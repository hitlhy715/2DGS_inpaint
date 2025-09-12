import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from mmengine import Config
import wandb
from tqdm import tqdm
from lpips import LPIPS
import math
import cv2
import warnings
warnings.filterwarnings('ignore')

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.distributed import init_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LambdaLR
from transformers import get_constant_schedule_with_warmup

from src.datasets.dataset import inpaint_dataset, RatioDistributedSampler
from src.gaussian import patch_gaussian
from src.loss.taming_dis import weights_init, NLayerDiscriminator as ldm_D, gradient_penalty_loss
from src.utils import util
from src.dino_cls_token import create_model as create_dino_model, get_feature as get_dino_feature


def ddp_setup(rank: int, world_size: int):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12356"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cosine_similarity(feature1, feature2):
    assert feature1.shape == feature2.shape

    f1 = F.normalize(feature1, dim=-1)
    f2 = F.normalize(feature2, dim=-1)

    cosine_sim = -1*(f1*f2).sum(dim=-1)
    return cosine_sim.mean()


def delay_lambda(current_step):
    return 0.0 


class Trainer:
    def __init__(self, cfg, device='cuda', dtype='fp32', rank=0, world_size=1):
        self.device = torch.device(device)
        self.dtype = util.to_torch_dtype(dtype)
        self.rank = rank
        self.world_size = world_size
        self.model_type = cfg.get('model_type', 'ae')
        self.model_condition_type = cfg['model_param']['condition_type']
        
        train_param = cfg['train_param']
        self.model_param = cfg['model_param']

        # init wandb
        self.wandb = False
        if train_param['wandb'] and self.rank==0:
            self.wandb = True
            wandb.init(project=train_param['wandb_proj'], name=train_param['wandb_name'])
        
        self.enable_mask = train_param['mask']


        # mask ratio
        self.use_multi_mask_ratio = train_param['use_multi_mask_ratio']
        self.ratio_interp = train_param['ratio_interp']
        self.min_ratio = train_param['min_ratio']
        self.max_ratio = train_param['max_ratio']
        
        # load dataset
        dataset_param = cfg['data_param']
        shuffle = dataset_param.pop('shuffle')
        num_workers = dataset_param.pop('num_workers')
        batch_size = dataset_param.pop('bs')
        dataset = inpaint_dataset(**dataset_param, min_ratio=self.min_ratio, max_ratio=self.max_ratio)

        if self.use_multi_mask_ratio:
            sampler = RatioDistributedSampler
        else:
            sampler = DistributedSampler

        self.dataloader = DataLoader(dataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size, sampler=sampler(dataset))
        self.data_iter = len(self.dataloader)


        self.use_dino_cls = self.model_param.get('use_dino_cls', False)
        self.use_dino_pred_loss = self.model_param.get('use_dino_pred_loss', False)

        if self.use_dino_cls:
            self.dino_model = create_dino_model().to(self.device, self.dtype)
            util.freeze(self.dino_model)
        else:
            self.dino_model=None

        # init model
        if self.model_type == 'ae':
            self.model = patch_gaussian(**cfg['model_param'], is_train=True, dino_model=self.dino_model).to(self.device, self.dtype)
        else:
            raise NotImplementedError
    

        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)


        # init param
        self.lr = train_param['lr']
        self.global_step = 0
        self.start_epoch = 0
        self.end_epoch = train_param['epoch']
        self.log_step = train_param['log_step']
        self.save_step = train_param['save_step']
        self.sample_step = train_param['sample_step']
        self.sample_dir = os.path.join(train_param['sample_dir'], train_param['exp_name'])
        self.output_dir = os.path.join(train_param['output_dir'], train_param['exp_name'])
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Generator trainable params: {util.format_numel(sum(p.numel() for p in self.model.parameters() if p.requires_grad))}")


        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=1e-3)
        self.scheduler = MultiStepLR(self.optimizer, milestones=train_param['milestones'], gamma=0.2, last_epoch=self.start_epoch-1)
        self.loss_type = train_param['loss_type']


        if self.loss_type.get('gan', None):
            self.discriminator = ldm_D(norm_type = 'gn').apply(weights_init).to(self.device, self.dtype)
            self.discriminator = DDP(self.discriminator, device_ids=[self.rank])
            self.gan_iter_step = self.loss_type['gan_iter_step']

            self.d_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=train_param['gan_lr'], weight_decay=1e-3)

           
            if train_param['d_warmup_step'] == 0 and  train_param['d_zero_step']==0:
                self.d_scheduler =  MultiStepLR(self.d_optimizer, milestones=train_param['milestones'], gamma=0.5, last_epoch=self.start_epoch-1)
                self.d_zero_step = -1
            else:
                delay_scheduler = LambdaLR(self.d_optimizer, lr_lambda=delay_lambda)
                warmup_scheduler = get_constant_schedule_with_warmup(
                    self.d_optimizer, 
                    num_warmup_steps=train_param['d_warmup_step'], 
                )
                self.d_scheduler = SequentialLR(
                    self.d_optimizer,
                    schedulers=[delay_scheduler, warmup_scheduler],
                    milestones=[train_param['d_zero_step']]
                )
                self.d_zero_step = train_param['d_zero_step']
            
            print(f"Discriminator trainable params: {util.format_numel(sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad))}")
        else:
            print('No Discriminator')


        # resume
        if train_param['resume']:
            checkpoint = train_param['model_path']
            assert checkpoint is not None
            self.load(checkpoint)
        
        
        assert (self.end_epoch-self.start_epoch) >0

        # loss related
        if self.loss_type['reconstruction']:
            if self.loss_type['reconstruction'].endswith('l1'):
                self.reconstruction_loss = nn.L1Loss()
            elif self.loss_type['reconstruction'].endswith('l2'):
                self.reconstruction_loss = nn.MSELoss()

        
        self.percep_model = None
        if self.loss_type['perceptual_loss'] == 'lpips':
            self.percep_model = LPIPS().to(self.device, self.dtype)
            util.freeze(self.percep_model)


    
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']

        self.model.module.load_state_dict(ckpt['model'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

        if self.loss_type.get('gan', None):
            missing_keys, unexpected_keys = self.discriminator.module.load_state_dict(ckpt['d_model'], strict=False)
            if unexpected_keys:
                raise RuntimeError(f'Discriminator Unexpected keys:{unexpected_keys}')
            if missing_keys:
                print(f'Discriminator Missing keys:{missing_keys}')

            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.d_scheduler.load_state_dict(ckpt['d_scheduler'])


    def update_gan(self, pred, gt, dino_feature=None):
        util.unfreeze(self.discriminator)
        util.freeze(self.model)


        gt.requires_grad_(True).to(self.discriminator.device)

        for i in range(self.gan_iter_step):
            
            self.d_optimizer.zero_grad()
            
            # Real
            D_real = self.discriminator(gt)
            # fake
            D_fake = self.discriminator(pred)


            if self.loss_type['gan_gradient_penalty']:
                gradient_penalty = gradient_penalty_loss(self.discriminator, gt, pred) * (self.loss_type['r1_weight'] //2)
            else:
                gradient_penalty = torch.zeros(1).to(D_real.device)

            # loss for discriminator
            if self.loss_type.get('gan', None) == 'hinge':
                D_loss = F.relu(1-D_real).mean() + F.relu(1+D_fake).mean()
            elif self.loss_type.get('gan', None) == 'logistic':
                D_loss = (F.softplus(-D_real).mean() + F.softplus(D_fake).mean())*0.5 
            else:
                raise NotImplementedError

            D_loss = D_loss + gradient_penalty

            D_loss.backward(retain_graph=((i < self.gan_iter_step - 1)))
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1) 

            self.gradient_penalty = gradient_penalty
            self.D_loss = D_loss
            self.d_optimizer.step()


        util.freeze(self.discriminator)
        util.unfreeze(self.model)
    

    def get_gan_loss(self, pred, dino_feature=None):
        D_fake = self.discriminator(pred)

        if self.loss_type.get('gan', None) == 'hinge':
            gan_loss = -torch.mean(D_fake)
        elif self.loss_type.get('gan', None) == 'logistic':
            gan_loss = F.softplus(-D_fake).mean()
        else:
            raise NotImplementedError

        return gan_loss



    def update(self, pred, gt, mask, dino_feature=None, dino_pred_fea=None, dino_clean_feature=None):
        self.optimizer.zero_grad()

        self.dino_pred_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.use_dino_cls:
            assert (dino_pred_fea is not None) and (dino_clean_feature is not None)
            self.dino_pred_loss = cosine_similarity(dino_pred_fea, dino_clean_feature)
        self.dino_pred_loss = self.dino_pred_loss * self.loss_type['dino_pred_scale']


        self.gan_loss_n = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.loss_type.get('gan', None) and (self.epoch_id>self.d_zero_step) and self.global_step>500:
            self.update_gan(pred.detach(), gt.clone(), dino_feature=dino_feature)
            self.gan_loss_n = self.get_gan_loss(pred, dino_feature=dino_feature)
        self.gan_loss_n = self.gan_loss_n*self.loss_type['gan_scale']


        self.recons_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.loss_type['reconstruction']:
            if self.loss_type['reconstruction'].startswith('full'):
                self.recons_loss = self.reconstruction_loss(pred, gt)
            elif self.loss_type['reconstruction'].startswith('mask'):
                self.recons_loss = self.reconstruction_loss(pred*(1-mask), gt*(1-mask))
        self.recons_loss = self.recons_loss * self.loss_type['recons_scale']

        
        self.perceptual_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.loss_type['perceptual_loss']:
            self.perceptual_loss = self.percep_model(pred, gt).mean()
        self.perceptual_loss = self.perceptual_loss * self.loss_type['perceptual_scale']


        self.loss = self.gan_loss_n + self.recons_loss + self.perceptual_loss + self.dino_pred_loss
        self.loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

        self.optimizer.step()
        
      
    
    def sample(self, preds, masks, gts, img_paths, epoch):
        bs = preds.shape[0]
        savedir = os.path.join(self.sample_dir, f'Epoch{epoch}_Step{self.global_step}')
        os.makedirs(savedir, exist_ok=True)
        preds = torch.clamp(preds, 0,1)

        for b in range(bs):
            pred = (preds[b]*255).to(dtype=torch.uint8)
            gt = (gts[b]*255).to(dtype=torch.uint8)
            mask = masks[b]
            mask_input = (gt*mask).to(dtype=torch.uint8)

            img_path = img_paths[b]
            img_name = os.path.splitext(os.path.basename(img_path))[0] 
            savep = os.path.join(savedir, f'sample_{img_name}.jpg')

            if self.enable_mask:
                out_img = torch.cat([pred, mask_input, gt], dim=-1)
            else:
                out_img = torch.cat([pred, gt], dim=-1)

            cv2.imwrite(savep, out_img.permute(1,2,0).numpy()[:,:,::-1])



    def train(self):
        log_loss = 0
        print(f'Training from epoch {self.start_epoch} to epoch {self.end_epoch}, each with {self.data_iter//self.world_size} iterations')

        for epoch_id in range(self.start_epoch, self.end_epoch+1):
            self.epoch_id = epoch_id
            print(f"[GPU{self.rank}] Epoch {epoch_id} | Steps: {len(self.dataloader)}")
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch_id}')
            self.dataloader.sampler.set_epoch(epoch_id)

            if self.use_multi_mask_ratio:
                ratio = self.min_ratio + 0.1*(epoch_id // self.ratio_interp)
                ratio = self.max_ratio if ratio>self.max_ratio else ratio

                print(f'current mask ratio:', ratio)
                self.dataloader.sampler.set_ratio(ratio, self.min_ratio, self.max_ratio)


            for iter_id, batch in enumerate(pbar):
                img = batch['img'].to(self.device, self.dtype)  # [B 3 H W]
                img_path = batch['img_path'] # list of length B

                mask = batch['mask'].to(self.device, self.dtype)
                dino_feature = None
                dino_clean_feature = None
                if self.enable_mask:
                    input = img*mask       
                    input_sets = (input, mask)   

                    if self.use_dino_cls:   
                        dino_feature = get_dino_feature(input, self.dino_model)   
                        if self.use_dino_pred_loss:
                            dino_clean_feature = get_dino_feature(img, self.dino_model)
                    
                else:
                    input_sets = (img,)


                dino_pred_fea=None
                pred, dino_pred_fea = self.model(*input_sets, dino_cls_token=dino_feature)

                if isinstance(pred, torch.Tensor):
                    pred = torch.clamp(pred, 0, 1)
                elif isinstance(pred, list):
                    pred = [torch.clamp(_,0,1) for _ in pred]

        
                self.update(pred, img, mask, dino_feature=dino_feature, dino_pred_fea=dino_pred_fea, dino_clean_feature=dino_clean_feature)
                log_loss += self.loss.item()


                pbar.set_postfix({
                        "loss":self.loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'd_lr':self.d_optimizer.param_groups[0]['lr'] if hasattr(self, 'd_optimizer') else 0,
                        "recons_loss":self.recons_loss.item(), 
                        "gan_loss":self.gan_loss_n.item() if self.loss_type.get('gan', None) else 0, 
                        "gradient_penalty":self.gradient_penalty.item() if hasattr(self, 'gradient_penalty') else 0, 
                        "dino_pred_loss": self.dino_pred_loss.item() if self.use_dino_pred_loss else 0,
                        "perceptual_loss":self.perceptual_loss.item(), 
                        "D_loss":self.D_loss.item() if hasattr(self, 'D_loss') else 0, 
                        })

                if self.rank == 0:
                    if self.global_step%self.log_step == 0:
                        avg_loss = log_loss / self.log_step
                        log_loss = 0
                        if self.wandb:
                            wandb_dict={
                                "loss":self.loss.item(),
                                "dino_pred_loss": self.dino_pred_loss.item(),
                                "gan_loss":self.gan_loss_n.item(),
                                "recons_loss":self.recons_loss.item(),
                                "D_loss":self.D_loss.item() if hasattr(self, 'D_loss') else 0,
                                "perceptual_loss":self.perceptual_loss.item(),
                                "gradient_penalty": self.gradient_penalty.item() if hasattr(self, 'gradient_penalty') else 0,
                                "avg_loss":avg_loss,
                                "global_step":self.global_step,
                                "lr":self.optimizer.param_groups[0]["lr"],
                                "d_lr":self.d_optimizer.param_groups[0]['lr'] if hasattr(self, 'd_optimizer') else 0,
                            }
                            wandb.log(wandb_dict, step=self.global_step)
                    
                    if self.global_step % self.sample_step == 0:
                        self.sample(pred.detach().cpu(), mask.detach().cpu(), img.detach().cpu(), img_path, epoch_id)

                    if self.global_step%self.save_step == 0:
                        ckpt_p = os.path.join(self.output_dir, f'epoch{epoch_id}_step{self.global_step}.pth')
                        ckpt_dict = dict(
                            epoch = epoch_id,
                            global_step = self.global_step,
                            model=self.model.module.state_dict(),
                            optimizer = self.optimizer.state_dict(),
                            scheduler = self.scheduler.state_dict(),
                            d_model = self.discriminator.module.state_dict() if self.loss_type.get('gan') else None,
                            d_optimizer = self.d_optimizer.state_dict() if self.loss_type.get('gan') else None,
                            d_scheduler = self.d_scheduler.state_dict() if self.loss_type.get('gan') else None,
                        )
                        torch.save(ckpt_dict, ckpt_p)

                self.global_step += 1

            self.scheduler.step()
            if hasattr(self, 'd_scheduler'):
                self.d_scheduler.step()

            pbar.close()

    

def train(cfg_path, **kwargs):
    cfg = Config.fromfile(cfg_path)
    pipe = Trainer(cfg, **kwargs)
    if cfg['model_type'] == 'ae':
        pipe.train()
    else:
        raise NotImplementedError
    

def mp_train(rank, world_size, cfg_path):
    ddp_setup(rank, world_size)
    train(cfg_path, rank=rank, world_size=world_size)
    

def mp_train_wrapper(cfg_path):
    world_size = torch.cuda.device_count()
    print(f'World size:{world_size}')
    if world_size >= 1:
        print(f'Train using {world_size} gpus')
        mp.spawn(mp_train, args=(world_size, cfg_path), nprocs=world_size)
    else:
        assert None, f'wrong world size:{world_size}'


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='train_args')
    parser.add_argument('--config_idx', type=str, default='', help='config file index')
    parser.add_argument('--config_path', type=str, default='', help='config file path')
    return parser.parse_args()


if __name__ =='__main__':
    parser = get_parser()

    if parser.config_idx !='':
        cfg_path = f'configs/train_exp{parser.config_idx}_cfg.py'
    elif parser.config_path !='':
        cfg_path = parser.config_path
    print(f'Using config file {cfg_path}')

    mp_train_wrapper(cfg_path)


