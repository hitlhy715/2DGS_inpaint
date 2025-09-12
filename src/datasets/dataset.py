import os
import random
import cv2
from PIL import Image, ImageFile

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from datasets import mask_gen


class inpaint_dataset(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None):
        self.mask_type = mask_type
        self.target_size = (256,256)
        self.ratio = None
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if os.path.isdir(path):
            if is_train:
                txt_path = os.path.join(path, 'train.txt')
            else:
                txt_path = os.path.join(path, 'test.txt')
        else:
            txt_path = path

        with open(txt_path, 'r') as f:
            self.imgs = [_.rstrip('\n') for _ in f.readlines()]

    
    def __len__(self):
        return len(self.imgs)
    
    def load_img(self, index):
        img_path = self.imgs[index]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img = transforms.ToTensor()(img)

        return img, img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
            
        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)


    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        img, img_path = self.load_img(index)
        mask = self.load_mask(img)
        return {'img': img, 'img_path': img_path, 'mask': mask}




class RatioDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, 
                 shuffle=True, seed=0, drop_last=False, ratio=0.5):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dataset = dataset
        self.ratio = ratio 
        
    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio = ratio
        self.dataset.set_ratio(ratio, min_ratio, max_ratio)
            

