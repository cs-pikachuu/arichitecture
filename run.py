import argparse
import logging
import os
from pathlib import Path
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate import FullyShardedDataParallelPlugin
from torchvision import transforms
import torch

# ours code base
from datasets.blended_mvs import BlendedMVS_Dataset
from utils.helper import sam2_forward, vggt_forward

import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
vggt_path = os.path.join(base_dir, "modules/vggt")
sys.path.insert(0, vggt_path)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
sys.path.pop(0)

sam2_path = os.path.join(base_dir, "modules/sam2")
sys.path.insert(0, sam2_path)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sys.path.pop(0)

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--vis_vggt",
        default=False,
        action="store_true",
        help="whether or not visualize vggt result.",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=10,
        help="Number of video frames.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=FullyShardedDataParallelPlugin(activation_checkpointing=False),
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # sam2
    sam2_checkpoint = ".modules/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = ".modules/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_predictor = SAM2ImagePredictor(build_sam2(sam2_model_cfg, sam2_checkpoint))
    sam2_predictor = sam2_predictor.to(accelerator.device)
    sam2_predictor.requires_grad_(False)
    
    # vggt
    vggt_model = VGGT()
    vggt_state_dict = torch.load("./modules/vggt/checkpoints/model.pt", map_location=accelerator.device)
    vggt_model.load_state_dict(vggt_state_dict, strict=False)
    vggt_model.eval()
    vggt_model = vggt_model.to(accelerator.device)
    vggt_model.requires_grad_(False)

    # Initialize the optimizer
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset_name is not None:
        if accelerator.is_main_process:
            print('************** Loading Dataset **************')
        train_dataset = BlendedMVS_Dataset(
            data_root=args.dataset_name,
            video_transforms=train_transforms,
            tokenizer=None,
            video_length=args.nframes
        )

    if accelerator.is_main_process:
        print('************** finish loading *************')
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    for batch in train_dataloader:
            images_list = []
            for single_img in batch['images']:
                images_list.append(single_img.squeeze(0).squeeze(1))

            if args.vis_vggt:
                output_dir = args.output_dir
            else:
                output_dir = None

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16): 
                masks = sam2_forward(
                    sam2_predictor,
                    images_list,
                    device=accelerator.device
                )
                vggt_forward(
                    vggt_model, 
                    pose_encoding_to_extri_intri,
                    unproject_depth_map_to_point_map,
                    images_list,
                    extra_seg_list=masks,
                    scene_name=batch["scene"][0],
                    vis_dir=output_dir,
                    device=accelerator.device
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()