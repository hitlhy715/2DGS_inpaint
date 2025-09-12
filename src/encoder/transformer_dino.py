import sys
from src.dinov2.models.vision_transformer import gs_vit_small, vit_base, gs_vit_small_map

import torch.nn as nn
from src.utils.block import MLP


class dinov2_encoder(nn.Module):
    def __init__(self, gaussian_per_patch, hidden_dim, image_size, patch_size, encoder_args, condition_type='direct', embed_dim=768, return_map_layer=[]):
        super().__init__()
        self.condition_type = condition_type
        self.return_map_layer = return_map_layer
        if condition_type == 'direct' or condition_type == 'multi_scale_direct':
            gs_hidden_dim = gaussian_per_patch*hidden_dim
            vit_hidden_dim = gs_hidden_dim//4

            # self.encoder = gs_vit_small(
            #     img_size=image_size[0],
            #     patch_size=patch_size[0],
            #     in_chans = 3,
            #     embed_dim=vit_hidden_dim,
            #     **encoder_args)

            self.encoder = gs_vit_small_map(
                img_size=image_size[0],
                patch_size=patch_size[0],
                in_chans = 3,
                embed_dim=vit_hidden_dim,
                return_map_layer=return_map_layer,
                **encoder_args)

            self.vit_mlp = MLP(
                in_dim = vit_hidden_dim,
                out_dim = gs_hidden_dim,
                hidden_list=[gs_hidden_dim, gs_hidden_dim, gs_hidden_dim]
            )
        elif self.condition_type == 'condition':
            self.encoder = vit_base(
                img_size=image_size[0],
                patch_size=patch_size[0],
                in_chans = 3,
                embed_dim=embed_dim,
                **encoder_args
            )
    
    def forward(self, img, mask=None):
        if self.return_map_layer:
            feat, attn_map = self.encoder(img, mask)
        else:
            feat = self.encoder(img, mask)

        if self.condition_type == 'direct' or self.condition_type == 'multi_scale_direct':
            feat = self.vit_mlp(feat)
        
        if self.return_map_layer:
            return feat, attn_map
        else:
            return feat