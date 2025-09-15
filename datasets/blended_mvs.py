import os
import re
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as tr

def default_loader(path):
    Im = Image.open(path)
    return Im.convert('RGB')


class BlendedMVS_Dataset(data.Dataset):
    def __init__(self,
        data_root,
        video_transforms=None,
        **kargs
        ):

        super().__init__()
        self.loader = default_loader
        self.img_transform = video_transforms
        self.scenes = self._make_dataset(data_root)

        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        scene = self.scenes[index]

        frames, segs, poses, intrinsics, depths = self.load_and_transform_frames(scene, self.loader, self.img_transform)

        frames = torch.cat(frames, 1) # c,t,h,w
        frames = frames.transpose(0, 1) # t,c,h,w
        
        example = dict()
        example["images"] = frames
        example["pose"] = poses
        example["intrinsic"] = intrinsics
        example["depth"] = depths

        return example

    def __len__(self):
        return len(self.scenes) 
    
    def _make_dataset(self, data_root):
        if not os.path.exists(data_root):
            raise RuntimeError("Dataset root directory does not exist: {}".format(data_root))
        pid_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        scenes = []
        for pid in pid_dirs:
            pid_root = os.path.join(data_root, pid)
            img_dir = os.path.join(pid_root, 'blended_images')
            cam_dir = os.path.join(pid_root, 'cams')
            depth_dir = os.path.join(pid_root, 'rendered_depth_maps')
            if not (os.path.isdir(img_dir) and os.path.isdir(cam_dir) and os.path.isdir(depth_dir)):
                print(f"Warning: Missing required directories for PID {pid}. Skipping this PID.")
                continue

            img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') and not f.endswith('_masked.jpg')]
            frame_ids = []
            for f in img_files:
                m = re.match(r'^(\d{8})\.jpg$', f)
                if m:
                    frame_ids.append(m.group(1))
            frame_ids.sort(key=lambda x: int(x))

            scene = []
            for fid in frame_ids:
                frame = {
                    "image": os.path.join(img_dir, f"{fid}.jpg"),
                    "mask_image": os.path.join(img_dir, f"{fid}_masked.jpg"),
                    "depth": os.path.join(depth_dir, f"{fid}.pfm"),
                    "cam": os.path.join(cam_dir, f"{fid}_cam.txt"),
                    "pair_txt": os.path.join(cam_dir, "pair.txt"),
                    "pid": pid,
                    "fid": fid,
                }
                if not (os.path.isfile(frame["image"]) and os.path.isfile(frame["cam"]) and os.path.isfile(frame["depth"])):
                    print(f"Warning: Missing required files for frame ID {fid} in PID {pid}. Skipping this frame.")
                    continue
                scene.append(frame)

            if len(scene) > 0:
                scenes.append(scene)

        return scenes

    def load_and_transform_frames(self, scene, loader, img_transform=None):
        clip = []
        pose_clip = []
        intrinsic_clip = []
        depth_clip = []
        seg_clip = []

        for frame in scene:
            # gt image
            fpath = frame["image"]
            #960,384
            img = loader(fpath)
            img_w, img_h = img.size[:2]

            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
                
            img = img.view(img.size(0),1, img.size(1), img.size(2))

            can_load = True

            # depth

            # pose and intrinsic

            # segmentation

            #load all
            if can_load:
                #image
                clip.append(img)
                # seg
                seg_clip.append(1)            
                # pose
                pose_clip.append(1)
                #intrinsic
                intrinsic_clip.append(1)    
                #depth            
                depth_clip.append(1)


        return clip, seg_clip, pose_clip, intrinsic_clip, depth_clip