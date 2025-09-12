import math
import numpy as np
import torch
import torch.nn as nn


from src.feature_map.direct_map import DirectMap
from src.encoder.transformer_dino import dinov2_encoder
from src.encoder.resnet_lama import ResNetEncoder
from src.utils.overlap import overlap_rasterize
from src.utils.block import MLP
from src.gaussian_kernel_bigger_patch import project_bigger_patch_gs, rasterize_bigger_patch_gs


class patch_gaussian(nn.Module):
    def __init__(self, 
            image_size, 
            patch_size, 
            gaussian_per_patch, 
            hidden_dim, 
            encoder_args, 
            encoder_type = 'vit',
            gs_fix=False, 
            overlap=False, 
            overlap_pad=1, 
            output_size=None,
            condition_type='direct',
            embed_dim=768, # only works for condition type is condition
            blk_num=6, # only works for condition type is condition
            return_map_layer=[],

            is_train = True,
            use_dino_cls = False,
            dino_model=None,
            dino_feature_dim=768,
            use_dino_pred_loss = False,
        )->None:
        super(patch_gaussian, self).__init__()

        self.return_map_layer = return_map_layer
        self.condition_type = condition_type
        self.is_train = is_train
        
        # feature alignment
        self.use_dino_cls = use_dino_cls
        self.use_dino_pred_loss = use_dino_pred_loss
        if self.use_dino_cls:
            self.dino_model = dino_model
            if self.use_dino_pred_loss:
                self.dino_map_mlp = MLP(
                    in_dim=dino_feature_dim, 
                    out_dim = dino_feature_dim, 
                    hidden_list = [int(dino_feature_dim*4)])


        self.hidden_dim = hidden_dim
        self.gaussian_per_patch = gaussian_per_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_num = (
            (self.image_size[0] + self.patch_size[0] - 1) // self.patch_size[0],
            (self.image_size[1] + self.patch_size[1] - 1) // self.patch_size[1],
        )


        encoder_args['use_dino_cls'] = use_dino_cls
        if encoder_type == 'vit':
            self.encoder = dinov2_encoder(self.gaussian_per_patch, hidden_dim, self.image_size, self.patch_size, encoder_args, condition_type=condition_type, embed_dim=embed_dim, return_map_layer=return_map_layer, use_dino_cls=use_dino_cls)
        elif encoder_type == 'resnet':
            self.encoder = ResNetEncoder(self.gaussian_per_patch, hidden_dim, self.patch_size, encoder_args)


        # when input size and output size is different
        self.out_patch_size = self.patch_size
        self.out_image_size = self.image_size
        if output_size is not None:
            assert (output_size[0] % self.patch_num[0] == 0) and (output_size[1]%self.patch_num[1]==0), f'output size {output_size} not match patch number{self.patch_num}'
            p1 = output_size[0]//self.patch_num[0]
            p2 = output_size[1]//self.patch_num[1]
            self.out_patch_size = (p1,p2)
            self.out_image_size = output_size

        # overlap
        self.overlap = overlap
        self.overlap_pad = overlap_pad
        if self.overlap:
            self.out_patch_size = (self.out_patch_size[0]+2*overlap_pad, self.out_patch_size[1]+2*overlap_pad)
            self.out_image_size = (self.out_image_size[0]+2*overlap_pad*self.patch_num[0], self.out_image_size[1]+2*overlap_pad*self.patch_num[1])
    

        if condition_type == 'direct':
            self.feature_map = DirectMap(hidden_dim, gaussian_per_patch, self.out_patch_size, self.out_image_size, self.image_size, gs_fix=gs_fix)


    @property
    def get_device(self):
        return next(self.encoder.parameters()).device
    
    def gen_feat(self, inp, dino_cls_token=None):
        """Generate feature by encoder."""
        feat = self.encoder(inp, None, dino_cls_token=dino_cls_token)
        return feat
    

    def forward_base(self, x, mask=None, dino_cls_token=None):
        '''
        x: [B 3 H W]
        mask: [B 1 H W]
        '''

        if self.use_dino_pred_loss:
            dino_cls_token = self.dino_map_mlp(dino_cls_token)

        if self.return_map_layer:
            feat, attn_map = self.gen_feat(x, dino_cls_token=dino_clean_feature) # e.g [B,256, gaussian*C]
        else:
            feat = self.gen_feat(x, dino_cls_token=dino_cls_token) # e.g [B,256, gaussian*C]

        bs,_, H,W = x.shape

        self.feature_map.map(feat, bs)

        pred = []
        for i in range(bs):
            get_xyz, weighted_cholesky, color_, opacity = self.feature_map.get_iter(i)

            xys, conics, = project_bigger_patch_gs(get_xyz, weighted_cholesky, self.out_patch_size, self.patch_num)
            img = rasterize_bigger_patch_gs(xys, conics, color_, opacity, self.out_image_size, self.out_patch_size, self.patch_num)

            img = img.contiguous()

            if self.overlap:
                img = overlap_rasterize(img, patch_size=self.out_patch_size, patch_num=self.patch_num, overlap_pad=self.overlap_pad)

            pred.append(img)
        pred = torch.stack(pred)
        pred = pred.permute(0,3,1,2)

        if self.is_train:
            if self.use_dino_pred_loss:
                return pred, dino_cls_token
            else:
                return pred
        else:
            return pred


    def forward(self, *args, **kwargs):
        if self.condition_type == 'direct':
            return self.forward_base(*args, **kwargs)

