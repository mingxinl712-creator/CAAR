import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import av

class MetaWorld_Dataset(Dataset):
    def __init__(
            self,
            args,
    ):
        super().__init__()
        self.args = args
        annotation_path = args.annotation_path
        data_root_path = args.data_root_path
        self.video_path = []
        data_json_path = f'{annotation_path}/data.json'
        with open(data_json_path, "r") as f:
            self.samples = json.load(f)
        self.video_path = [os.path.join(data_root_path, sample['dataset_name']) for sample in self.samples]

        self.use_augmentation = args.use_augmentation

        self.basic_transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.aug_transform = T.Compose([
            T.RandomResizedCrop(448, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC), 
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor()
        ])

        self.max_views = args.max_views

        stats_json_path = f'{annotation_path}/stats.json'
        with open(stats_json_path, "r") as f:
            stats = json.load(f)
        self.s_min = np.array(stats["state"]["min"])
        self.s_max = np.array(stats["state"]["max"])
        self.a_min = np.array(stats["action"]["min"])
        self.a_max = np.array(stats["action"]["max"])

    def __len__(self):
        return len(self.samples)
    
    def _get_frames(self, label, frame_id, cam_id, video_dir):
        video_path = label['videos'][cam_id]['video_path']
        video_path = os.path.join(video_dir, video_path)

        frames = []
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)

        image = frames[frame_id]        # (H,W,C), uint8
        image = Image.fromarray(image)
        if self.use_augmentation:
            image = self.aug_transform(image) if random.random() < 0.5 else self.basic_transform(image)
        else:
            image = self.basic_transform(image)
        return image

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def process_action(self, label, frame_ids, action_mask):
        frame_ids = frame_ids[:int(self.args.num_frames)]
        states = np.array(label['states'])[frame_ids]
        actions = np.array(label['actions'])[frame_ids]
        action_mask = np.array(action_mask)
        actions = actions * action_mask[:, None]
        state = states[0:1] # current state

        # normalize
        action_scaled = self.normalize_bound(actions, self.a_min, self.a_max)
        state_scaled = self.normalize_bound(state, self.s_min, self.s_max)
        return torch.from_numpy(action_scaled).float(), torch.from_numpy(state_scaled).float()

    def __getitem__(self, index):
        sample = self.samples[index]
        sampled_video_dir = self.video_path[index]

        ann_file = sample['ann_file']
        ann_file = f'{sampled_video_dir}/{ann_file}'
        frame_ids = sample['frame_ids']
        with open(ann_file, "r") as f:
            label = json.load(f)

        data = dict()
        # action
        action_mask = sample['mask']
        data['actions'], data['states'] = self.process_action(label, frame_ids, action_mask)
        # instructions
        data['prompts'] = label['tasks'][0]
        # observation
        image = self._get_frames(label, frame_ids[0], cam_id=0, video_dir=sampled_video_dir)
        images = []
        images.append(image)
        num_real_views = len(images)
        image_masks = torch.zeros(self.max_views, dtype=torch.bool)
        image_masks[:num_real_views] = True
        while len(images) < self.max_views:
            dummy_image = torch.zeros_like(images[0])
            images.append(dummy_image)
        images = torch.stack(images)
        data['images'] = images
        data['image_masks'] = image_masks

        return {
            "images": data['images'],
            "image_mask": data['image_masks'],
            "prompts": data['prompts'],
            "states": data['states'],
            "actions": data['actions'],
        }
