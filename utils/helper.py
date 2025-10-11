import numpy as np
import matplotlib.cm as cm
import imageio
import torch
import os
from tqdm import tqdm
from IPython import embed
import laspy
from pdb import set_trace
from PIL.ImageOps import exif_transpose
from PIL import Image
import torchvision.transforms as tvf
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from time import time, strftime, localtime
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import cv2
from typing import List, Tuple
from save import save_video, save_images, visualize_poses, _resize_pil_image, load_images, save_las, enhanced_visualize_tracks

def _has_masks(seg) -> bool:
    """安全判断 seg 是否包含任何可用的 mask。"""
    if seg is None:
        return False
    if isinstance(seg, np.ndarray):
        # 单个 mask：只要有非零像素就认为有
        return seg.size > 0 and np.any(seg)
    if torch.is_tensor(seg):
        return seg.numel() > 0 and bool(seg.any().item())
    if isinstance(seg, (list, tuple)):
        return len(seg) > 0
    return False

def _normalize_seg_to_items(seg, fallback_id):
    """
    把 seg 统一成 [(mask2d_like, (box_id,)), ...] 列表；mask 可以是 np.ndarray 或 torch.Tensor。
    fallback_id 用于没 box_id 时构造一个稳定 id。
    """
    items = []
    if seg is None:
        return items

    # 旧格式：[(mask, (box_id,)), ...]
    if isinstance(seg, (list, tuple)) and len(seg) > 0 and isinstance(seg[0], (list, tuple)):
        for it in seg:
            if len(it) == 0:
                continue
            mask = it[0]
            meta = it[1] if len(it) >= 2 else (fallback_id,)
            if isinstance(meta, (list, tuple)) and len(meta) >= 1:
                box_id = meta[0]
            else:
                box_id = fallback_id
            items.append((mask, (box_id,)))
        return items

    # 单个 ndarray / tensor
    if isinstance(seg, np.ndarray) or torch.is_tensor(seg):
        items.append((seg, (fallback_id,)))
        return items

    # 新格式：List[np.ndarray / tensor]
    if isinstance(seg, (list, tuple)):
        for j, m in enumerate(seg):
            items.append((m, (fallback_id * 1000 + j,)))
        return items

    return items

def project_points_with_depth_ordering(pts_and_colors, 
                                    img_board,
                                    extrinsics, 
                                    intrinsics, 
                                    H, W, orig_H, orig_W,
                                    device=torch.device('cpu')):
    """
    支持两种输入格式的投影函数：
    1. 背景点云：{'points': [1,N,3], 'colors': [1,N,3]}
    2. 前景点云：{box_id: {'points': [T,Ni,3], 'colors': [T,Ni,3]}}
    
    对于前景点云，会自动按时间帧匹配相机投影（前3张相机用时间帧1的点云...）
    """
    if isinstance(extrinsics, list):
        extrinsics = torch.stack(extrinsics, dim=0).to(device)  # [num_cams, 4, 4]
    else:
        extrinsics = extrinsics.to(device)
        
    if isinstance(intrinsics, list):
        intrinsics = torch.stack(intrinsics, dim=0).to(device)  # [num_cams, 3, 3]
    else:
        intrinsics = intrinsics.to(device)

    dtype = torch.float64
    extrinsics = extrinsics.to(dtype)
    intrinsics = intrinsics.to(dtype)
    
    num_cams = extrinsics.shape[0]
    num_time_frames = num_cams // 3

    if 'points' in pts_and_colors: 
        all_points = pts_and_colors['points'].squeeze(0).to(dtype)  # [N, 3]
        all_colors = pts_and_colors['colors'].squeeze(0)  # [N, 3]
        
        for cam_idx in range(num_cams):
            pts_hom = torch.cat([all_points, torch.ones_like(all_points[:, :1])], dim=-1)
            pts_cam = (pts_hom @ extrinsics[cam_idx].T)[:, :3]
            
            z = pts_cam[:, 2]
            valid = z > 1e-3
            u = intrinsics[cam_idx, 0, 0] * (pts_cam[:, 0]/z) + intrinsics[cam_idx, 0, 2]
            v = intrinsics[cam_idx, 1, 1] * (pts_cam[:, 1]/z) + intrinsics[cam_idx, 1, 2]
            
            u = u.round().long()
            v = v.round().long()
            valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            
            if valid.any():
                u_valid = u[valid]
                v_valid = v[valid]
                colors_valid = all_colors[valid]
                
                depth_order = torch.argsort(z[valid])
                img_board[cam_idx].index_put_(
                    (v_valid[depth_order], u_valid[depth_order]), 
                    colors_valid[depth_order], 
                    accumulate=False
                )

    else:  # {box_id: {'points': [T,Ni,3], 'colors': [T,Ni,3]}}
        for time_idx in range(num_time_frames):
            cam_start = time_idx * 3
            cam_end = cam_start + 3

            combined_points = []
            combined_colors = []
            
            for box_id, data in pts_and_colors.items():
                if time_idx < len(data['points']):
                    frame_points = data['points'][time_idx].squeeze(0)  # [Ni, 3]
                    frame_colors = data['colors'][time_idx].squeeze(0)  # [Ni, 3]
                    combined_points.append(frame_points)
                    combined_colors.append(frame_colors)
            
            if not combined_points:
                continue
                
            all_points = torch.cat(combined_points, dim=0).to(dtype)  # [N_total, 3]
            all_colors = torch.cat(combined_colors, dim=0)  # [N_total, 3]

            for cam_idx in range(cam_start, min(cam_end, num_cams)):
                pts_hom = torch.cat([all_points, torch.ones_like(all_points[:, :1])], dim=-1)
                pts_cam = (pts_hom @ extrinsics[cam_idx].T)[:, :3]
                
                z = pts_cam[:, 2]
                valid = z > 1e-3
                u = intrinsics[cam_idx, 0, 0] * (pts_cam[:, 0]/z) + intrinsics[cam_idx, 0, 2]
                v = intrinsics[cam_idx, 1, 1] * (pts_cam[:, 1]/z) + intrinsics[cam_idx, 1, 2]
                
                u = u.round().long()
                v = v.round().long()
                valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                
                if valid.any():
                    depth_order = torch.argsort(z[valid])
                    u_sorted = u[valid][depth_order]
                    v_sorted = v[valid][depth_order]
                    colors_sorted = all_colors[valid][depth_order]

                    img_board[cam_idx].index_put_(
                        (v_sorted, u_sorted),
                        colors_sorted,
                        accumulate=False
                    )
    
    return img_board

def adjust_depth_scale(depth_map, extra_depth):
    adjusted_depth_map = []

    for i in range(len(extra_depth)):
        current_depth_map = depth_map[i]  # [H, W, 1]
        current_extra_depth = extra_depth[i]  # [1, H', W']

        current_depth_map = current_depth_map.squeeze(-1)  # [H, W]

        depth_min = current_depth_map.min()
        depth_max = current_depth_map.max()
        extra_min = current_extra_depth.min()
        extra_max = current_extra_depth.max()

        scale_factor = (extra_max - extra_min) / (depth_max - depth_min)

        adjusted_depth = (current_depth_map - depth_min) * scale_factor + extra_min

        adjusted_depth = adjusted_depth.unsqueeze(-1)  # [H, W, 1]

        adjusted_depth_map.append(adjusted_depth)

    adjusted_depth_map = torch.stack(adjusted_depth_map, dim=0)  # [N, H', W', 1]
    return adjusted_depth_map

def extend_pose(pose):
    N = pose.shape[0]
    last_row = torch.tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device).repeat(N, 1, 1)
    extended_pose = torch.cat([pose, last_row], dim=1)
    return extended_pose

def apply_masks_to_images(images, seg_list, alpha=0.5):
    """
    Args:
        images: Tensor of shape [1, N, 3, H_new, W_new]
        seg_list: 每帧的分割；允许以下任一格式：
            - None
            - np.ndarray(H, W) 单个mask
            - List[np.ndarray(H, W)] 多个mask
            - List[(mask, (box_id,))] 旧格式
        alpha: blending factor (0 = original, 1 = full color)
    Returns:
        colored_images: Tensor of shape [N, 3, H_new, W_new]
    """
    import numpy as np

    device = images.device
    color_palette = torch.tensor([
        [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [1,0,1], [0,1,1],
        [1,0.5,0], [0.5,0,1], [0,0.5,1],
        [0.5,1,0], [1,0,0.5], [0.5,0.5,1]
    ], device=device, dtype=torch.float32)

    images = images.squeeze(0)  # [N, 3, H, W]
    N, C, H_new, W_new = images.shape
    colored_images = []

    def _normalize_frame_seg(seg_data, frame_idx):
        """把任意 seg_data 统一成 List[(torch.bool mask_2d, (box_id,))]."""
        norm = []
        if seg_data is None:
            return norm
        # 单个 np.ndarray
        if isinstance(seg_data, np.ndarray):
            m = seg_data
            if m.ndim != 2:
                m = np.squeeze(m)
                if m.ndim != 2:
                    return norm
            if m.dtype == np.uint8:
                m_bool = m > 0
            elif np.issubdtype(m.dtype, np.floating):
                m_bool = m > 0.5
            else:
                m_bool = m.astype(bool, copy=False)
            m_t = torch.from_numpy(m_bool).to(device=device)
            norm.append((m_t, (frame_idx,)))
            return norm

        # 列表/元组
        if isinstance(seg_data, (list, tuple)):
            # 旧格式：[(mask, (box_id,)), ...]
            if len(seg_data) > 0 and isinstance(seg_data[0], (list, tuple)) and len(seg_data[0]) >= 1:
                for item in seg_data:
                    mask = item[0]
                    box_meta = item[1] if (len(item) >= 2) else (frame_idx,)
                    if isinstance(mask, np.ndarray):
                        if mask.ndim != 2:
                            mask = np.squeeze(mask)
                            if mask.ndim != 2:
                                continue
                        if mask.dtype == np.uint8:
                            m_bool = mask > 0
                        elif np.issubdtype(mask.dtype, np.floating):
                            m_bool = mask > 0.5
                        else:
                            m_bool = mask.astype(bool, copy=False)
                        m_t = torch.from_numpy(m_bool).to(device=device)
                    elif torch.is_tensor(mask):
                        m_t = mask.to(device=device).bool()
                        if m_t.ndim != 2:
                            m_t = m_t.squeeze()
                            if m_t.ndim != 2:
                                continue
                    else:
                        continue
                    # 规范化 box_meta 成 (box_id,)
                    if isinstance(box_meta, (list, tuple)) and len(box_meta) >= 1:
                        box_id = box_meta[0]
                    else:
                        box_id = frame_idx
                    norm.append((m_t, (box_id,)))
                return norm
            else:
                # 新格式：List[np.ndarray]
                for j, m in enumerate(seg_data):
                    if isinstance(m, np.ndarray):
                        if m.ndim != 2:
                            m = np.squeeze(m)
                            if m.ndim != 2:
                                continue
                        if m.dtype == np.uint8:
                            m_bool = m > 0
                        elif np.issubdtype(m.dtype, np.floating):
                            m_bool = m > 0.5
                        else:
                            m_bool = m.astype(bool, copy=False)
                        m_t = torch.from_numpy(m_bool).to(device=device)
                        norm.append((m_t, (frame_idx * 1000 + j,)))
                    elif torch.is_tensor(m):
                        m_t = m.to(device=device).bool()
                        if m_t.ndim != 2:
                            m_t = m_t.squeeze()
                            if m_t.ndim != 2:
                                continue
                        norm.append((m_t, (frame_idx * 1000 + j,)))
                return norm

        # 其它不支持的类型，返回空
        return norm

    for img_idx, (img_tensor, seg_data) in enumerate(zip(images, seg_list)):
        colored_img = img_tensor.clone()  # [3, H_new, W_new]

        seg_items = _normalize_frame_seg(seg_data, img_idx)  # List[(mask2d_bool, (box_id,))]
        if len(seg_items) == 0:
            colored_images.append(colored_img)
            continue

        for mask_bool, (box_id,) in seg_items:
            # 尺寸对齐到 [H_new, W_new]
            if mask_bool.shape != (H_new, W_new):
                mask_bool = mask_bool.float().unsqueeze(0).unsqueeze(0)
                mask_bool = F.interpolate(mask_bool, size=(H_new, W_new), mode='nearest').squeeze().bool()
            # 取颜色
            color_idx = int(hash(box_id)) % len(color_palette)
            color = color_palette[color_idx].view(3, 1, 1)  # [3,1,1]
            colored_mask = color.expand(-1, H_new, W_new)  # [3, H_new, W_new]
            # 混合
            mask_idx = mask_bool
            colored_img[:, mask_idx] = (1 - alpha) * colored_img[:, mask_idx] + alpha * colored_mask[:, mask_idx]

        colored_images.append(colored_img)

    return torch.stack(colored_images, dim=0)  # [N, 3, H_new, W_new]

def split_point_cloud_and_colors(images, point_maps, seg_list):
    """
    images: torch.Tensor，形状 [1, N, 3, H, W] 或 [N, 3, H, W]
    point_maps: 可索引的序列，长度为总视角数；每个元素形状 [H, W, 3]（torch/numpy 都可）
    seg_list: 长度与视角数对齐或更短；元素可为
              None / np.ndarray(2D) / torch.Tensor(2D) /
              List[np.ndarray/torch.Tensor] /
              旧格式 List[(mask, (box_id,)), ...]
    返回：
      bg_result: {'points': [1, K, 3], 'colors': [1, K, 3]}
      fg_data:   {box_id: {'points': [T*[1, Ni, 3]], 'colors': [T*[1, Ni, 3]]}}
    """
    device = images.device

    # ---- 统一 images 形状：支持 5D 或 4D ----
    if images.dim() == 5:
        if images.size(0) != 1:
            raise ValueError(f"Expect images [1,N,3,H,W] when 5D, got {tuple(images.shape)}")
        images = images.squeeze(0)  # -> [N,3,H,W]
    elif images.dim() == 4:
        pass  # 已是 [N,3,H,W]
    else:
        raise ValueError(f"images must be 4D or 5D, got {tuple(images.shape)}")

    N, C, H, W = images.shape

    # ---- 统一 point_maps 为可索引 tensor 列表 ----
    pm_list = []
    if torch.is_tensor(point_maps):
        # 允许传 [N,H,W,3] 的整块 tensor
        if point_maps.dim() != 4 or point_maps.shape[-1] != 3:
            raise ValueError(f"point_maps expected [N,H,W,3], got {tuple(point_maps.shape)}")
        for i in range(point_maps.shape[0]):
            pm_list.append(point_maps[i])
    elif isinstance(point_maps, (list, tuple)):
        for i, pm in enumerate(point_maps):
            if isinstance(pm, np.ndarray):
                pm_t = torch.from_numpy(pm)
            elif torch.is_tensor(pm):
                pm_t = pm
            else:
                raise TypeError(f"point_maps[{i}] must be ndarray or tensor, got {type(pm)}")
            if pm_t.dim() != 3 or pm_t.shape[-1] != 3:
                raise ValueError(f"point_maps[{i}] expected [H,W,3], got {tuple(pm_t.shape)}")
            pm_list.append(pm_t.to(device=device, dtype=torch.float32))
    else:
        raise TypeError(f"point_maps must be tensor or list/tuple, got {type(point_maps)}")

    total_views = len(pm_list)
    if total_views == 0:
        empty = torch.empty(1, 0, 3, device=device)
        return {'points': empty, 'colors': empty}, {}

    # ---- 动态判断每帧视角数 ----
    if total_views % 3 == 0 and total_views // 3 >= 1:
        cams_per_frame = 3
        num_frames = total_views // 3
    else:
        cams_per_frame = 1
        num_frames = total_views

    seg_len = len(seg_list) if isinstance(seg_list, (list, tuple)) else 0

    # ---- 统计跨帧持久的 box_id（可选）----
    persistent_ids = set()
    if num_frames > 1 and seg_len > 0:
        frame_id_sets = [set() for _ in range(num_frames)]
        for time_idx in range(num_frames):
            frame_start = time_idx * cams_per_frame
            frame_end = min(frame_start + cams_per_frame, seg_len)
            for idx in range(frame_start, frame_end):
                s = seg_list[idx]
                if not _has_masks(s):
                    continue
                items = _normalize_seg_to_items(s, fallback_id=idx)
                for _, (box_id,) in items:
                    frame_id_sets[time_idx].add(box_id)
        if frame_id_sets:
            persistent_ids = set(frame_id_sets[0])
            for t in range(1, num_frames):
                persistent_ids.intersection_update(frame_id_sets[t])

    bg_points = []
    bg_colors = []
    fg_data = {}  # {box_id: {'points': [T * [1,Ni,3]], 'colors': [T * [1,Ni,3]]}}

    # ---- 主循环：按帧聚合 ----
    for time_idx in range(num_frames):
        frame_start = time_idx * cams_per_frame
        frame_end = min(frame_start + cams_per_frame, total_views)

        temp_bg_points = []
        temp_bg_colors = []

        for idx in range(frame_start, frame_end):
            pts_hw3 = pm_list[idx]  # [H,W,3]
            if pts_hw3.device != device:
                pts_hw3 = pts_hw3.to(device=device)
            pts = pts_hw3.reshape(-1, 3)  # [H*W, 3]

            img = images[idx].permute(1, 2, 0).reshape(-1, 3).contiguous()  # [H*W,3], float

            seg_data = seg_list[idx] if idx < seg_len else None
            if _has_masks(seg_data):
                # 初始：全部视作背景，随后从前景里剔除
                bg_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=device)
                items = _normalize_seg_to_items(seg_data, fallback_id=idx)

                for mask, (box_id,) in items:
                    # 统一 2D bool mask -> (H,W)
                    if isinstance(mask, np.ndarray):
                        m = mask
                        if m.ndim != 2:
                            m = np.squeeze(m)
                        if m.ndim != 2:
                            continue
                        if m.dtype == np.uint8:
                            m_bool = m > 0
                        elif np.issubdtype(m.dtype, np.floating):
                            m_bool = m > 0.5
                        else:
                            m_bool = m.astype(bool, copy=False)
                        m_t = torch.from_numpy(m_bool).to(device=device)
                    elif torch.is_tensor(mask):
                        m_t = mask.to(device=device).bool()
                        if m_t.ndim != 2:
                            m_t = m_t.squeeze()
                            if m_t.ndim != 2:
                                continue
                    else:
                        continue

                    if m_t.shape != (H, W):
                        m_t = F.interpolate(
                            m_t.float().unsqueeze(0).unsqueeze(0),
                            size=(H, W), mode='nearest'
                        ).squeeze().bool()

                    mask_flat = m_t.view(-1)
                    masked_pts = pts[mask_flat]
                    masked_cols = img[mask_flat]

                    # 收集持久前景
                    if box_id in persistent_ids:
                        if box_id not in fg_data:
                            fg_data[box_id] = {
                                'points': [[] for _ in range(num_frames)],
                                'colors': [[] for _ in range(num_frames)]
                            }
                        fg_data[box_id]['points'][time_idx].append(masked_pts.unsqueeze(0))
                        fg_data[box_id]['colors'][time_idx].append(masked_cols.unsqueeze(0))

                    bg_mask = bg_mask & ~mask_flat

                # 背景剩余
                temp_bg_points.append(pts[bg_mask].unsqueeze(0))
                temp_bg_colors.append(img[bg_mask].unsqueeze(0))
            else:
                # 没有分割：整张都当背景
                temp_bg_points.append(pts.unsqueeze(0))
                temp_bg_colors.append(img.unsqueeze(0))

        # 这一帧的 BG 聚合；兜底避免空列表
        if len(temp_bg_points) > 0:
            bg_points.append(torch.cat(temp_bg_points, dim=1))  # [1, sum_i, 3]
            bg_colors.append(torch.cat(temp_bg_colors, dim=1))
        else:
            empty = torch.empty(1, 0, 3, device=device)
            bg_points.append(empty)
            bg_colors.append(empty)

    # ---- 把 FG 的时间维串起来 ----
    for box_id in list(fg_data.keys()):
        pts_frames = []
        col_frames = []
        for t in range(num_frames):
            pts_list = fg_data[box_id]['points'][t]
            col_list = fg_data[box_id]['colors'][t]
            if len(pts_list) > 0:
                pts_frames.append(torch.cat(pts_list, dim=1))  # [1, Ni, 3]
                col_frames.append(torch.cat(col_list, dim=1))
            else:
                empty = torch.empty(1, 0, 3, device=device)
                pts_frames.append(empty)
                col_frames.append(empty)
        fg_data[box_id]['points'] = pts_frames
        fg_data[box_id]['colors'] = col_frames

    # ---- 最终 BG 串起来（时间维 concat） ----
    if len(bg_points) > 0:
        bg_points_cat = torch.cat(bg_points, dim=1)  # [1, total_bg, 3]
        bg_colors_cat = torch.cat(bg_colors, dim=1)
    else:
        empty = torch.empty(1, 0, 3, device=device)
        bg_points_cat = empty
        bg_colors_cat = empty

    bg_result = {'points': bg_points_cat, 'colors': bg_colors_cat}
    return bg_result, fg_data

import numpy as np

def convert_seg_list_to_coords(seg_list, hw_wh):
    """
    把各种可能的 seg_list 统一转换为：
      coords_list: List[np.ndarray (N_i, 2)]  # 每个点为 (x, y)
      boxes_list:  List[np.ndarray (4,)]      # [x1, y1, x2, y2]
    兼容输入形式：
      - 单个 mask: np.ndarray(H,W)
      - 多个 mask: List[np.ndarray(H,W)]
      - 旧格式:    List[ (mask, meta) ]，其中 meta 可忽略
    hw_wh: (H, W)
    """
    H, W = (int(hw_wh[0]), int(hw_wh[1]))

    # 统一包装为 list
    if isinstance(seg_list, np.ndarray):
        segs = [seg_list]
    elif isinstance(seg_list, (list, tuple)):
        segs = list(seg_list)
    else:
        raise TypeError(f"seg_list must be ndarray or list/tuple, got {type(seg_list)}")

    coords_list, boxes_list = [], []

    for i, item in enumerate(segs):
        # 取出 mask
        if isinstance(item, np.ndarray):
            mask = item
        elif isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], np.ndarray):
            mask = item[0]  # 旧格式 (mask, meta...) -> 只用 mask
        else:
            raise TypeError(f"seg_list[{i}] must be ndarray or (ndarray, ...), got {type(item)}")

        # squeeze 到 2D
        mask = np.asarray(mask)
        if mask.ndim != 2:
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"seg_list[{i}] expected 2D mask, got shape {mask.shape}")

        # 统一成 bool
        if mask.dtype == np.uint8:
            m_bool = mask > 0
        elif np.issubdtype(mask.dtype, np.floating):
            m_bool = mask > 0.5
        else:
            m_bool = mask.astype(bool, copy=False)

        # 提取点坐标（注意 xy 顺序）
        ys, xs = np.nonzero(m_bool)
        if xs.size == 0:
            coords = np.zeros((0, 2), dtype=np.float32)
            box = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
        else:
            coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2) : (x,y)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            box = np.array([x1, y1, x2, y2], dtype=np.float32)

        coords_list.append(coords)
        boxes_list.append(box)

    return coords_list, boxes_list


def vggt_forward(
    model, 
    pose_encoding_to_extri_intri, 
    unproject_depth_map_to_point_map, 
    images_list, 
    extra_seg_list=None,
    scene_name="noname",
    vis_dir=None, 
    device=torch.device('cpu')):

    C, orig_H, orig_W = images_list[0].shape  # 保留你的原写法

    images = load_images(
        images_list,
        device=device
    )
    all_frame_num, C, H, W = images.shape   
    frame_num = len(images_list) 

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            images = images[None]  # add batch dimension: [1, T, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extrinsic = extrinsic.squeeze(0)
        intrinsic = intrinsic.squeeze(0)

        extrinsic = extend_pose(extrinsic) # w2c

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        depth_map = depth_map.squeeze(0)

        # ---------- 构造 query_points（改动点 1/2） ----------
        # 当没有有效 mask 时，传空的 torch 张量，避免后续 .clone() 报错
        m0 = extra_seg_list[0] if (extra_seg_list is not None and len(extra_seg_list) > 0) else None
        if (m0 is None) or (isinstance(m0, np.ndarray) and not np.any(m0)):
            query_points_t = torch.zeros(1, 0, 2, device=device, dtype=torch.float32)  # (B=1, N=0, 2)
            track_list = torch.zeros(1, all_frame_num, 0, 2, device=device)
            vis_score  = torch.zeros(1, all_frame_num, 0, device=device)
            conf_score = torch.zeros(1, all_frame_num, 0, device=device)
        else:
            # 用你的转换函数把 seg -> 点集（numpy）
            coords_list, boxes_list = convert_seg_list_to_coords(extra_seg_list, (H, W))

            # 取第一个实例的点作为 query；没有就用中心点兜底
            if len(coords_list) == 0 or coords_list[0].shape[0] == 0:
                pts = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)  # (1,2)
            else:
                pts = coords_list[0].astype(np.float32)                 # (N,2)
                # 可选：若点太多，做下采样
                if pts.shape[0] > 512:
                    idx = np.linspace(0, pts.shape[0] - 1, 512, dtype=int)
                    pts = pts[idx]

            # numpy -> torch (B=1, N, 2) 放到 device（改动点 2/2）
            query_points_t = torch.from_numpy(pts).to(device=device, dtype=torch.float32).unsqueeze(0)

            # 如果你的 tracker 期望 0~1 归一化坐标，可解除下面两行注释：
            # query_points_t[..., 0] /= (W - 1)
            # query_points_t[..., 1] /= (H - 1)

            # 进入追踪头
            track_list, vis_score, conf_score = model.track_head(
                aggregated_tokens_list, images, ps_idx,
                query_points=query_points_t   # torch.Tensor (1,N,2)
            )
            track_list = track_list[-1]

            # 可视化（把 query_points 再转回 numpy，形状 (N,2)）
            if vis_dir is not None:
                enhanced_visualize_tracks(
                    images=images,                    # [1, T, 3, H, W]
                    tracks=track_list,                # [1, T, N, 2] torch
                    vis_scores=vis_score,             # [1, T, N]    torch
                    conf_scores=conf_score,           # [1, T, N]    torch
                    # query_points=query_points_t[0].detach().cpu().numpy(),  # (N,2) numpy
                    query_points=query_points_t,
                    output_dir=vis_dir,
                    vis_threshold=0.2,
                    conf_threshold=0.2,
                    cmap_name="rainbow",
                    frames_per_row=6,
                    device=device
                )
        
        # Construct 3D Points from Depth Maps and Cameras
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map, extrinsic, intrinsic
        )
        point_map_by_unprojection = torch.from_numpy(point_map_by_unprojection).to(device)

    # Aggregate colors
    images = apply_masks_to_images(images, extra_seg_list)
    bkg_points, frg_points_dict = split_point_cloud_and_colors(
        images, point_map_by_unprojection, extra_seg_list
    )

    img_board = [torch.zeros(H, W, 3, device=device) for _ in range(frame_num)]
    img_board = project_points_with_depth_ordering(
        bkg_points, img_board, extrinsic, intrinsic, H, W, orig_H, orig_W, device=device
    )
    img_board = project_points_with_depth_ordering(
        frg_points_dict, img_board, extrinsic, intrinsic, H, W, orig_H, orig_W, device=device
    )
    # Resize
    for i in range(len(img_board)):
        img_board[i] = img_board[i].permute(2, 0, 1).unsqueeze(0)
        img_board[i] = F.interpolate(img_board[i], size=(orig_H, orig_W), mode='nearest')

    if vis_dir is not None:
        timestamp = time()
        scene_folder = os.path.join(vis_dir, f"{scene_name}_{timestamp}")
        src_folder = os.path.join(scene_folder, "src")
        os.makedirs(src_folder, exist_ok=True)
        seged_folder = os.path.join(scene_folder, "seged")
        os.makedirs(seged_folder, exist_ok=True)

        # ✅ 构造原图序列（NHWC, uint8）的 numpy 数组，避免 torch.stack 报错
        try:
            if isinstance(images_list[0], torch.Tensor):
                # 若你的 images_list 已经是 NCHW Tensor
                src_arr = torch.stack(images_list, dim=0).permute(0, 2, 3, 1).detach().cpu().numpy()
            else:
                # 常见情况：images_list 是一组 HWC uint8 的 numpy
                # 确保都是 HWC，再堆叠成 NHWC
                src_arr = np.stack(
                    [img if img.ndim == 3 else np.squeeze(img) for img in images_list],
                    axis=0
                )
                # 若不是 uint8，可按需转：src_arr = src_arr.astype(np.uint8, copy=False)
        except Exception as e:
            # 兜底：逐张转成 numpy HWC 再堆叠
            tmp = []
            for im in images_list:
                if isinstance(im, torch.Tensor):
                    tmp.append(im.permute(1, 2, 0).detach().cpu().numpy())
                else:
                    tmp.append(np.asarray(im))
            src_arr = np.stack(tmp, axis=0)

        # 保存原图与叠加可视化
        save_images(src_arr, src_folder)  # 直接 NHWC numpy
        save_images(images.permute(0, 2, 3, 1).detach().cpu().numpy(), seged_folder)

        # 保存点云（背景）
        save_las([bkg_points['points']], [bkg_points['colors']], os.path.join(src_folder, f"pts.las"))

        print(f"Saved vis in {src_folder}, {seged_folder}")


def sam2_forward(
    sam2_predictor,
    images_list,           # list[np.ndarray], HWC uint8 RGB
    device: str = "cuda",
    multimask_output: bool = False,
):
    """
    使用 SAM2ImagePredictor 做一次“盒提示”推理：
      - 不再传入 point_coords/point_labels（避免断言）
      - 仅传入覆盖整幅图像的 box，当作一个通用的初始提示
    返回：与 images_list 等长的 mask 列表（每个元素为 np.ndarray，H×W，bool 或 uint8）
    """
    results = []
    for img in images_list:
        # 1) 设定图像（必须是 uint8 RGB，HWC）
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expect HWC RGB image, got {img.shape} with dtype={img.dtype}")

        sam2_predictor.set_image(img)  # 注意：predictor 自己持有设备上的模型，不需要 .to()

        H, W = img.shape[:2]
        # 2) 用整图 box 提示（注意 shape 要是 (N,4)）
        box = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)

        # 3) 只给 box，不给 point_coords/point_labels
        masks, scores, logits = sam2_predictor.predict(
            "segment different parts of the building, for example: roof, door ...",
            box=box,
            point_coords=None,
            point_labels=None,
            multimask_output=multimask_output,
        )
        # masks: (N, H, W) -> 取第一张
        m = masks[0]
        # 统一转成 uint8(0/255) 或 bool，按你后续需求，这里给 uint8
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8) * 255
        results.append(m)

    return results
