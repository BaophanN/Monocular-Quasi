import mmcv
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qd3dt.datasets import build_dataloader, build_dataset
from qd3dt.models import build_detector

from collections import defaultdict
from functools import partial
from mmcv.parallel import DataContainer as DC
from torchvision.transforms import ToTensor 
from torch.utils.data import Dataset 

class CustomDashCamDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 img_scales=[(640, 480)],
                 flip_ratio=0.0, 
                 resize_keep_ratio=True
                 ): 
        self.root_dir = root_dir 
        self.image_dir = os.path.join(root_dir, "data/")
        self.pose_file = os.path.join(root_dir, 'data/pose.txt')
        self.img_scales = img_scales 
        self.flip_ratio = flip_ratio 
        self.resize_keep_ratio = resize_keep_ratio

        self.img_infos = self._load_pose_info()

    def _load_pose_info(self):
        img_infos = {}
        with open(self.pose_file, "r") as f: 
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        for i in range(0, len(data_lines), 2):
            # Expecting at least 10 tokens: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            line = data_lines[i]
            tokens = line.strip().split()
            if len(tokens) < 8: 
                continue
            img_id = tokens[0]
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            img_infos[img_id]({
                "img_id": img_id, # 000x from kitti 
                "pose": np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float32),
                "rotation": (qw, qx, qy, qz),
                "translation": (tx, ty, tz),
                "first_frame": (i == 0),  # Mark first frame
                "video_id": 0,  # Single video sequence ID
                "index": i  # Frame index
            })        
            return img_infos    

    def __len__(self):
        return len(self.img_infos)

    def img_transform(self, img, scale, flip, keep_ratio=True):
        """Resize, flip, and normalize an image."""
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, scale_factor = mmcv.imresize(img, scale, return_scale=True)
        
        if flip:
            img = mmcv.imflip(img)
        
        img_shape = img.shape[:2]
        pad_shape = img.shape
        return img, img_shape, pad_shape, scale_factor


    def prepare_test_img(self, idx): 
        img_info = self.img_infos[idx] 
        img = mmcv.imread(img_info['file_name'])
        if img is None: 
            print(f'Error: Image{img_info['file_name']} not found.')
            return None 
        def prepare_single(img, scale, flip):
                """Prepare a single image with transformation."""
                _img, img_shape, pad_shape, scale_factor = self.img_transform(img, scale, flip, self.resize_keep_ratio)
                _img = torch.tensor(_img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Convert to Tensor

                pose = img_info['pose']

                _img_meta = dict(
                    ori_shape=img.shape[:2],
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    pose=pose,
                    first_frame=img_info['first_frame'],
                    video_id=img_info['video_id'],
                    frame_id=img_info['index'],
                    img_info=img_info,
                    flip=flip
                )
                return _img, _img_meta           
        imgs, img_metas = [], [] 
        for scale in self.img_scales: 
            _img, _img_meta = prepare_single(img, scale, False) 
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True)) 
            if self.flip_ratio > 0: 
                _img, _img_meta = prepare_single(img, scale, True) 
                imgs.append(_img) 
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas, )
        return data 


config_path = ''
cfg = mmcv.Config.fromfile(config_path)
dataset = cfg.dataset.type
sampler = None 
batch_size = 1
imgs_per_gpu = 1 
num_workers = 4 

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=False)

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
model.eval()
outputs = defaultdict(list)
prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
coco_outputs = defaultdict(list)
pred_id = 0
for _, data in enumerate(data_loader):
    with torch.no_grad():
        result, use_3d_center = model(
            return_loss=False, rescale=True, pure_det=pure_det, **data)

    img_info = data['img_meta'][0].data[0][0]['img_info']

    if img_info.get(
            'is_key_frame') is not None and not img_info['is_key_frame']:
        prog_bar.update()
        continue

    outputs['bbox_results'].append(result['bbox_results'])
    outputs['track_results'].append(result['track_results'])
    if 'depth_results' in result:
        outputs['depth_results'].append(result['depth_results'])

    prog_bar.update()

