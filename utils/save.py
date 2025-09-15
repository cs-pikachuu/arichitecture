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
from typing import List, Tuple, Optional
import matplotlib.colors

def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False, seg_data=None):
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float()
        frames = frames.permute(0, 2, 3, 1)  # -> [N, H, W, C]
        frames = frames.numpy()

    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])

    id_to_color = {}
    colormap = np.array(cm.get_cmap("tab20").colors) 

    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            frame = frames[i].copy()
            h, w = frame.shape[:2]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
            if seg_data is not None:
                current_seg = seg_data[i]
                overlay = frame.copy()
                
                for j, (mask, (box_id,)) in enumerate(current_seg):
                    if box_id not in id_to_color:
                        color_idx = hash(box_id) % len(colormap)
                        id_to_color[box_id] = (colormap[color_idx] * 255).astype(np.uint8)
                    
                    color = id_to_color[box_id]
                    mask_np = mask.squeeze(0).cpu().numpy()
                    
                    if mask_np.shape != frame.shape[:2]:
                        mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0])) > 0.5
                    
                    alpha = 0.97
                    overlay[mask_np > 0.5] = (1 - alpha) * overlay[mask_np > 0.5] + alpha * color

                    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w_box, h_box = cv2.boundingRect(contours[0])
                        cv2.putText(overlay, box_id[:6], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)
                
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            writer.append_data(frame)

    writer.close()

def save_images(frames, output_folder, is_depths=False, grayscale=False):
    os.makedirs(output_folder, exist_ok=True)

    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            imageio.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), depth_vis)
    else:
        for i in range(frames.shape[0]):
            frame = frames[i].copy()
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), frame)

def visualize_poses(poses_list, save_path):
    c2w_poses = torch.stack(poses_list).detach().cpu().numpy()

    positions = c2w_poses[:, :3, 3]
    right_axes = c2w_poses[:, :3, 0]  # 右向轴（X）
    up_axes = c2w_poses[:, :3, 1]     # 上向轴（Y）
    forward_axes = c2w_poses[:, :3, 2] # 前向轴（Z）

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2], 
        'k-',  # 黑色实线
        linewidth=1, 
        alpha=0.5,  # 半透明
        label='Camera Path'
    )
    for i, (x, y, z) in enumerate(positions):
        ax.text(x, y, z, str(i), color='red', fontsize=8)  # 在点上标数字
    # 2. 绘制相机位置（散点）
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2], 
        c='b', 
        s=20,  # 点大小
        label='Camera Positions'
    )
    for i in range(0, len(positions)):
        # ax.quiver(
        #     positions[i, 0], positions[i, 1], positions[i, 2],
        #     right_axes[i, 0], right_axes[i, 1], right_axes[i, 2],
        #     length=0.05, arrow_length_ratio=0.1, color='r', label='Right (X)' if i == 0 else None
        # )
        # ax.quiver(
        #     positions[i, 0], positions[i, 1], positions[i, 2],
        #     up_axes[i, 0], up_axes[i, 1], up_axes[i, 2],
        #     length=0.05, arrow_length_ratio=0.1, color='g', label='Up (Y)' if i == 0 else None
        # )
        ax.quiver(
            positions[i, 0], positions[i, 1], positions[i, 2],
            forward_axes[i, 0], forward_axes[i, 1], forward_axes[i, 2],
            length=0.05, arrow_length_ratio=0.1, color='b', label='Forward (Z)' if i == 0 else None
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(pil_images, device='cpu'):
    images = []
    shapes = set()
    to_tensor = tvf.ToTensor()

    # First process all images and collect their shapes
    for img in pil_images:

        img = to_pil_image(img).convert("RGB")

        width, height = img.size
        new_width = 518

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img).to(device)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518

        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y : start_y + 518, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(pil_images) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def save_las(all_pts3d_world, colored, filename):
    points = []
    for i in all_pts3d_world:
        if i is not None:
            points.append(i)
    points = torch.cat(points, dim=1).squeeze(0).detach().cpu().numpy()
    colors = []
    for i in colored:
        if i is not None:
            colors.append(i)
    colors = torch.cat(colors, dim=1).squeeze(0).detach().cpu().numpy()

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    points = np.array(points)
    colors = np.array(colors)

    if colors.dtype == np.float64 or colors.dtype == np.float32:
        colors = (colors * 255).astype(np.uint8)

    header = laspy.LasHeader(point_format=2)
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    las.write(filename)

def enhanced_visualize_tracks(
    images: torch.Tensor,                        # [B, S, 3, H, W] 或 [B, S, H, W, 3]
    tracks: torch.Tensor,                        # [B, S, N, 2]
    vis_scores: Optional[torch.Tensor] = None,   # [B, S, N]
    conf_scores: Optional[torch.Tensor] = None,  # [B, S, N]
    query_points: Optional[torch.Tensor] = None, # [B, N_q, 2]
    output_dir: str = "enhanced_visualization",
    vis_threshold: float = 0.2,
    conf_threshold: float = 0.2,
    cmap_name: str = "hsv",
    query_color: Tuple[int, int, int] = (255, 0, 0),
    frames_per_row: int = 4,
    save_grid: bool = True,
    device='cpu'
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestep = time()
    
    images = images.squeeze(0) if images.dim() == 5 else images  # [S, ...]
    tracks = tracks.squeeze(0) if tracks.dim() == 4 else tracks  # [S, N, 2]
    
    S, N, _ = tracks.shape
    H, W = images.shape[-2], images.shape[-1]
    
    track_colors = generate_track_colors(
        tracks, 
        vis_mask=(vis_scores.squeeze(0) > vis_threshold) if vis_scores is not None else None,
        image_width=W,
        image_height=H,
        cmap_name=cmap_name
    )
    
    # 处理每帧图像
    frame_images = []
    for s in range(S):
        # 转换图像格式
        img = prepare_image(images[s], format="CHW" if images.dim() == 4 else "HWC")
        
        # 标注查询点（仅第一帧）
        if s == 0 and query_points is not None:
            for x, y in query_points.squeeze(0).cpu().numpy():
                cv2.circle(img, (int(x), int(y)), 3, query_color, -1)
        
        # 标注轨迹点
        valid_mask = get_valid_mask(
            vis_scores[0,s] if vis_scores is not None else None,
            conf_scores[0,s] if conf_scores is not None else None,
            vis_threshold,
            conf_threshold,
            device=device
        )
        
        for i in range(N):
            if valid_mask is None or valid_mask[i]:
                x, y = tracks[s, i].cpu().numpy()
                cv2.circle(img, (int(x), int(y)), 3, track_colors[i], -1)
        
        frame_images.append(img)
    
    # 保存网格
    if save_grid and len(frame_images) > 0:
        save_image_grid(frame_images, frames_per_row, os.path.join(output_dir, f"grid_{timestep}.png"))

def generate_track_colors(tracks, vis_mask=None, image_width=None, image_height=None, cmap_name="hsv"):
    """生成基于位置的轨迹颜色 (返回BGR格式的整数三元组)"""
    N = tracks.shape[1]
    colors = []
    
    for i in range(N):
        # 找到第一个可见帧
        first_vis = 0
        if vis_mask is not None:
            vis_frames = torch.where(vis_mask[:, i])[0]
            first_vis = vis_frames[0] if len(vis_frames) > 0 else 0
        
        x, y = tracks[first_vis, i].tolist()
        
        # 获取颜色 (确保在[0,1]范围)
        r, g, b = matplotlib.cm.get_cmap(cmap_name)((x/image_width + y/image_height)/2)[:3]
        r, g, b = max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b))
        
        # 转换为BGR整数格式 (OpenCV要求)
        color_bgr = (int(b*255), int(g*255), int(r*255))
        colors.append(color_bgr)
    
    return colors

def prepare_image(img_tensor, format="CHW"):
    """转换图像张量到OpenCV格式"""
    img = img_tensor.float().cpu().numpy()
    if format == "CHW":
        img = img.transpose(1, 2, 0)  # HWC
    
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def get_valid_mask(vis_scores, conf_scores, vis_thresh, conf_thresh, device='cpu'):
    """生成有效点掩码"""
    if vis_scores is None and conf_scores is None:
        return None
    mask = torch.ones(len(vis_scores), dtype=torch.bool, device=device)
    if vis_scores is not None:
        mask &= (vis_scores > vis_thresh)
    if conf_scores is not None:
        mask &= (conf_scores > conf_thresh)
    return mask

def save_image_grid(images, frames_per_row, output_path):
    """保存图像网格"""
    H, W = images[0].shape[:2]
    num_rows = (len(images) + frames_per_row - 1) // frames_per_row
    grid_img = np.zeros((H*num_rows, W*frames_per_row, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row, col = i // frames_per_row, i % frames_per_row
        grid_img[row*H:(row+1)*H, col*W:(col+1)*W] = img
    
    cv2.imwrite(output_path, grid_img)