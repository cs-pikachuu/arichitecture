#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import glob
import traceback
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

# ========== 项目内导入路径修正 ==========
_CUR_DIR = os.path.dirname(__file__)
_BASE_DIR = os.path.abspath(os.path.join(_CUR_DIR))
_MODULES_DIR = os.path.join(_BASE_DIR, "modules")

if _MODULES_DIR not in sys.path:
    sys.path.insert(0, _MODULES_DIR)

# 允许缺失 __init__.py 的模块被 import（可选）
for _pkg in ("vggt", "sam2"):
    _pkg_dir = os.path.join(_MODULES_DIR, _pkg)
    _init = os.path.join(_pkg_dir, "__init__.py")
    if os.path.isdir(_pkg_dir) and not os.path.exists(_init):
        try:
            open(_init, "a").close()
        except Exception:
            pass

# vggt
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# utils
_UTILS_DIR = os.path.join(_BASE_DIR, "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from helper import sam2_forward, vggt_forward  # utils/helper.py

# -----------------------
# 工具函数
# -----------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def ensure_file_exists(p: str, hint: str = ""):
    if not os.path.isfile(p):
        h = f" ({hint})" if hint else ""
        raise FileNotFoundError(f"Missing file: {p}{h}")


def list_images(dir_path: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    files.sort()
    return files


def discover_scenes(data_root: str) -> List[str]:
    """
    在 data_root 下查找包含至少 1 张图片的子文件夹，作为“场景”；
    如果 data_root 自己就直接放了图片，则把 data_root 当作一个场景。
    """
    data_root = os.path.abspath(data_root)
    subdirs = [
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]
    candidate_scenes = []

    # 根目录本身
    root_imgs = list_images(data_root)
    if len(root_imgs) > 0:
        candidate_scenes.append(data_root)

    # 子目录
    for sd in sorted(subdirs):
        imgs = list_images(sd)
        if len(imgs) > 0:
            candidate_scenes.append(sd)

    if not candidate_scenes:
        raise RuntimeError(f"No scenes with images found under: {data_root}")

    return candidate_scenes


def load_first_k_frames(scene_dir: str, k: int) -> List[str]:
    imgs = list_images(scene_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No images in scene dir: {scene_dir}")
    return imgs[:k]


def pil_to_uint8_rgb_array(p: str) -> np.ndarray:
    """
    读取路径 p 到 uint8 H×W×3 RGB numpy 数组（0..255）
    """
    with Image.open(p) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)  # HWC uint8
    return arr


# -----------------------
# SAM2 构建辅助（不改 SAM 源码；多策略重试 + 支持 pkg/file provider + 前缀 configs/）
# -----------------------
def _normalize_config_name_no_suffix(name: str) -> str:
    """去掉 .yaml/.yml 后缀"""
    if name.endswith(".yaml") or name.endswith(".yml"):
        return name.rsplit(".", 1)[0]
    return name


def _make_attempts_for_yaml_path(yaml_path: Path) -> List[Tuple[str, List[str]]]:
    """
    给定一个 *文件路径*（可为软链），生成若干 (config_name, overrides) 组合。
    关键点：
      - 不 .resolve()，保留你传入的“无点号”软链名（如 sam2_1/...）
      - 同时尝试包内前缀 'configs/...'
      - 同时给出点号->下划线的别名
    """
    if yaml_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expect a YAML file, got: {yaml_path}")

    # 不要 resolve，保留软链名
    configs_root = yaml_path.parent.parent          # .../modules/sam2/configs
    pkg_root     = configs_root.parent              # .../modules/sam2   (pkg://sam2)
    group        = yaml_path.parent.name            # e.g. "sam2_1"  (软链目录)
    base         = yaml_path.stem                   # e.g. "sam2_1_hiera_l" (软链文件名去后缀)

    group_alias  = group.replace(".", "_")
    base_alias   = base.replace(".", "_")

    attempts: List[Tuple[str, List[str]]] = []

    # === File provider: searchpath 指到 configs 根 / 组目录 ===
    # A1) configs 根 + group/base
    attempts.append((
        f"{group}/{base}",
        [f"hydra.searchpath=[file://{configs_root}]", "hydra.job.chdir=False"]
    ))
    # A2) 组目录 + base
    attempts.append((
        base,
        [f"hydra.searchpath=[file://{(configs_root / group)}]", "hydra.job.chdir=False"]
    ))

    # === PKG provider: config_module='sam2' 下，名字要从包根起，前缀必须带 'configs/' ===
    # B1) 包根 + configs/group/base
    attempts.append((
        f"configs/{group}/{base}",
        []  # 用 pkg provider，overrides 不必注入 file://
    ))

    # === 别名（点号->下划线），同时覆盖 file 与 pkg 两路 ===
    # C1) file provider (configs 根)
    attempts.append((
        f"{group_alias}/{base_alias}",
        [f"hydra.searchpath=[file://{configs_root}]", "hydra.job.chdir=False"]
    ))
    # C2) file provider (组目录)
    attempts.append((
        base_alias,
        [f"hydra.searchpath=[file://{(configs_root / group_alias)}]", "hydra.job.chdir=False"]
    ))
    # C3) pkg provider (包根 + configs/...)
    attempts.append((
        f"configs/{group_alias}/{base_alias}",
        []
    ))

    # === 兜底：把 searchpath 指到包根（少数 repo 如此生效） ===
    attempts.append((
        f"configs/{group_alias}/{base_alias}",
        [f"hydra.searchpath=[file://{pkg_root}]", "hydra.job.chdir=False"]
    ))

    return attempts


def _build_sam2_with_retries(sam2_cfg: str, sam2_ckpt: str):
    """
    多策略重试构建 SAM2：
      - 若传入文件路径：依次尝试多种组合（file/pkg provider 皆试）
      - 若传入配置名：尝试原名 + 带 'configs/' 前缀 + 点号->下划线 别名
    """
    # 调试：打印用户传入的原始 sam2_cfg
    print(f"[DEBUG] user sam2_cfg raw path: {sam2_cfg}")

    p = Path(sam2_cfg)
    attempts: List[Tuple[str, List[str]]] = []
    if p.exists() and p.is_file() and p.suffix in {".yaml", ".yml"}:
        attempts = _make_attempts_for_yaml_path(p)
    else:
        # 配置名：构造多组尝试（包含 'configs/' 前缀 和 别名）
        cfg_name = _normalize_config_name_no_suffix(sam2_cfg)
        cfg_name_alias = cfg_name.replace(".", "_")
        attempts = [
            (cfg_name, []),
            (f"configs/{cfg_name}", []),
            (cfg_name_alias, []),
            (f"configs/{cfg_name_alias}", []),
        ]

    last_err = None
    for i, (config_name, overrides) in enumerate(attempts, 1):
        print(f"[DEBUG] SAM2 compose attempt #{i}: config_name='{config_name}', overrides={overrides}")
        try:
            # 优先尝试支持 hydra_overrides_extra 的签名
            try:
                model = build_sam2(
                    config_name,
                    sam2_ckpt,
                    hydra_overrides_extra=overrides,
                    map_location="cpu",
                )
            except TypeError:
                # 兼容老签名（无 hydra_overrides_extra 参数）
                model = build_sam2(config_name, sam2_ckpt)
            print(f"[INFO] SAM2 compose OK with attempt #{i}")
            return model
        except Exception as e:
            print(f"[WARN] SAM2 compose failed on attempt #{i}: {e.__class__.__name__}: {e}")
            last_err = e
            continue

    # 仍失败：抛出，并提示如何核对目录
    raise RuntimeError(
        "Failed to compose SAM2 config after multi-attempts.\n"
        "请核对以下几件事：\n"
        "  1) 路径大小写、下划线/连字符是否与文件夹/文件完全一致；\n"
        "  2) 目标目录确实存在并可读（包含 YAML）；\n"
        "  3) 若 YAML 内有 defaults 引用其它组，searchpath 必须指向 '.../configs' 根；\n"
        "  4) 也可以把 --sam2_cfg 直接写为 'configs/<group>/<name>' 这种“配置名”。\n"
    ) from last_err


# -----------------------
# 推理主逻辑
# -----------------------
def run_infer(
    data_root: str,
    nframes: int,
    vis_dir: str = None,
    sam2_cfg: str = "/home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_ckpt: str = "/home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/checkpoints/sam2.1_hiera_large.pt",
    vggt_ckpt: str = "/home/pikachu/Project/Farscape/zrl/arichitecture/modules/vggt/checkpoints/model.pt",
    device: str = None,
):
    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")

    # 路径存在性检查
    ensure_file_exists(sam2_ckpt, "SAM2 checkpoint")
    ensure_file_exists(vggt_ckpt, "VGGT checkpoint")

    # ========== 1) SAM2 ==========
    print("[INFO] Building SAM2...")
    sam2_model = _build_sam2_with_retries(sam2_cfg, sam2_ckpt)
    sam2_model.eval()
    # 把模型放到 device（predictor 不是 nn.Module，不能 .to）
    sam2_model.to(device)
    for p in sam2_model.parameters():
        p.requires_grad = False
    sam2_predictor = SAM2ImagePredictor(sam2_model)  # 不 .to()

    # ========== 2) VGGT ==========
    print("[INFO] Building VGGT...")
    vggt_model = VGGT().to(device)
    state = torch.load(vggt_ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # 去掉可能的 "module." 前缀
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    vggt_model.load_state_dict(state, strict=False)
    vggt_model.eval()
    vggt_model.requires_grad_(False)

    # ========== 3) 场景枚举 ==========
    scenes = discover_scenes(data_root)
    print(f"[INFO] Found {len(scenes)} scene(s) under: {data_root}")

    # ========== 4) 推理 ==========
    torch.set_grad_enabled(False)
    for sid, scene_dir in enumerate(scenes, 1):
        scene_name = os.path.basename(os.path.normpath(scene_dir))
        print(f"[INFO] [{sid}/{len(scenes)}] Scene: {scene_name}")

        img_paths = load_first_k_frames(scene_dir, nframes)
        images_list = [pil_to_uint8_rgb_array(p) for p in img_paths]  # list of HWC uint8

        # 可视化输出目录
        scene_vis_dir = None
        if vis_dir is not None:
            scene_vis_dir = os.path.join(vis_dir, scene_name)
            os.makedirs(scene_vis_dir, exist_ok=True)

        try:
            with torch.inference_mode():
                # (1) SAM2 前向：得到跨帧 mask 列表（按你自己的 sam2_forward 实现）
                masks = sam2_forward(
                    sam2_predictor,
                    images_list,     # list[np.ndarray HWC uint8]
                    device=device
                )

                # (2) VGGT 前向：把图像与分割结果传入（按你自己的 vggt_forward 实现）
                vggt_forward(
                    vggt_model,
                    pose_encoding_to_extri_intri,
                    unproject_depth_map_to_point_map,
                    images_list,         # 若 vggt_forward 期望 torch.Tensor，可在内部转换
                    extra_seg_list=masks,
                    scene_name=scene_name,
                    vis_dir=scene_vis_dir,
                    device=device
                )

        except Exception as e:
            print(f"[ERROR] Inference failed on scene '{scene_name}': {e}")
            traceback.print_exc()
            # 不中断，继续下一个场景
            continue

    print("[INFO] Inference finished.")


# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pure Inference: SAM2 + VGGT on (Blended)MVS-style data.")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing scenes or images.")
    parser.add_argument("--nframes", type=int, default=1, help="Number of frames per scene to process.")
    parser.add_argument("--vis_vggt", action="store_true", help="If set, save VGGT visualizations/outputs.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Base output dir (used when --vis_vggt).")
    parser.add_argument("--sam2_cfg", type=str, default="/home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default="/home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--vggt_ckpt", type=str, default="/home/pikachu/Project/Farscape/zrl/arichitecture/modules/vggt/checkpoints/model.pt")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto")
    return parser.parse_args()


def main():
    args = parse_args()

    vis_dir = None
    if args.vis_vggt:
        vis_dir = args.output_dir
        os.makedirs(vis_dir, exist_ok=True)

    run_infer(
        data_root=args.data_root,
        nframes=args.nframes,
        vis_dir=vis_dir,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        vggt_ckpt=args.vggt_ckpt,
        device=args.device
    )


if __name__ == "__main__":
    main()
