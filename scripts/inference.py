import os
import sys
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmengine import Config
import wandb
from tqdm import tqdm
import cv2

from src.datasets.dataset import inpaint_dataset
from src.gaussian import patch_gaussian
from src.utils.util import to_torch_dtype, freeze, unfreeze, set_random_seed
from src.dino_cls_token import create_model as create_dino_model, get_feature as get_dino_feature



class Inferencer:
    def __init__(self, cfg, device='cuda', dtype='fp32'):
        set_random_seed()
        self.device = torch.device(device)
        self.dtype = to_torch_dtype(dtype)
        self.condition_type = cfg['model_param']['condition_type']
        
        inference_param = cfg['inference_param']

        self.enable_mask = inference_param['mask']

        self.use_dino_cls = cfg['model_param'].get('use_dino_cls', False)
        self.dino_model=None
        if self.use_dino_cls:
            self.dino_model = create_dino_model().to(self.device, self.dtype)
            freeze(self.dino_model)
            
        
        # load dataset
        dataset_param = cfg['data_param']
        shuffle = dataset_param.pop('shuffle')
        num_workers = dataset_param.pop('num_workers')
        batch_size = dataset_param.pop('bs')

        dataset = inpaint_dataset(**dataset_param)
        self.dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        self.data_iter = len(self.dataloader)

        # init model
        self.model = patch_gaussian(**cfg['model_param'], dino_model=self.dino_model).to(self.device, self.dtype)

        # freeze model if needed
        freeze(self.model)

        # init param
        self.global_step = 0
        self.start_epoch = 0

        self.sample_dir = os.path.join(inference_param['sample_dir'], inference_param['exp_name'])
        os.makedirs(self.sample_dir, exist_ok=True)

        # resume
        assert isinstance(inference_param['model_path'], str) 
        checkpoint = inference_param['model_path']
        assert checkpoint is not None
        self.load(checkpoint)
        

    
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        if 'epoch' in ckpt.keys():
            self.start_epoch = ckpt['epoch']
            self.global_step = ckpt['global_step']

            self.model.load_state_dict(ckpt['model'])
        else:
            self.start_epoch = 'Unkown'
            self.global_step = ckpt_path.split('iter_')[-1].split('_net')[0]
            self.model.load_state_dict(ckpt)



    def sample(self, preds, masks, gts, img_paths):
        bs = preds.shape[0]
        savedir = os.path.join(self.sample_dir, f'Epoch{self.start_epoch}_Step{self.global_step}')
        os.makedirs(savedir, exist_ok=True)

        for b in range(bs):
            pred = preds[b]
            pred = torch.clamp(pred, 0., 1.)
            pred = (pred*255).to(dtype=torch.uint8)
            gt = (gts[b]*255).to(dtype=torch.uint8)
            mask = masks[b]
            mask_input = (gt*mask).to(dtype=torch.uint8)

            img_path = img_paths[b]
            img_name = os.path.splitext(os.path.basename(img_path))[0] 
            savep = os.path.join(savedir, f'sample_{img_name}.png')


            if self.enable_mask:
                out_img = torch.cat([pred, mask_input, gt], dim=-1)
                # out_img = pred
            else:
                # out_img = torch.cat([pred, gt], dim=-1)
                out_img = pred
                
            cv2.imwrite(savep, out_img.permute(1,2,0).numpy()[:,:,::-1])


    def inference(self):
        print(f'Inference {self.data_iter} iterations')
        pbar = tqdm(self.dataloader, desc=f'Inferencing')

        for iter, batch in enumerate(pbar):
            img = batch['img'].to(self.device, self.dtype)  # [B 3 H W]
            img_path = batch['img_path'] # list of length B
            mask = batch['mask'].to(self.device, self.dtype)

            dino_feature=None
            if self.enable_mask:
                input = img*mask
                if self.use_dino_cls:   
                    dino_feature = get_dino_feature(input, self.dino_model)  
                if self.condition_type == 'multi_scale_direct':
                    pred = self.model(input, mask, dino_cls_token=dino_feature)
                else:
                    pred = self.model(input, mask, dino_cls_token=dino_feature)
            else:
                input = img
                if self.use_dino_cls:   
                    dino_feature = get_dino_feature(input, self.dino_model)  
                pred = self.model(input, dino_cls_token=dino_feature)

            self.sample(pred.detach().cpu(), mask.detach().cpu(), img.detach().cpu(), img_path)


    

def inference(cfg_path):
    cfg = Config.fromfile(cfg_path)
    pipe = Inferencer(cfg)
    pipe.inference()


if __name__ =='__main__':
    inference('configs/inference_cfg.py')






