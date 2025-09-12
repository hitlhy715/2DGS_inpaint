# from: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
import abc
from functools import partial
import functools
import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class DepthWiseSeperableConv(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        if 'groups' in kwargs:
            # ignoring groups for Depthwise Sep Conv
            del kwargs['groups']
        
        self.depthwise = nn.Conv2d(in_dim, in_dim, *args, groups=in_dim, **kwargs)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class MultidilatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation_num=3, comb_mode='sum', equal_dim=True,
                 shared_weights=False, padding=1, min_dilation=1, shuffle_in_channels=False, use_depthwise=False, **kwargs):
        super().__init__()
        convs = []
        self.equal_dim = equal_dim
        assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
        if comb_mode in ('cat_out', 'cat_both'):
            self.cat_out = True
            if equal_dim:
                assert out_dim % dilation_num == 0
                out_dims = [out_dim // dilation_num] * dilation_num
                self.index = sum([[i + j * (out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
            else:
                out_dims = [out_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                out_dims.append(out_dim - sum(out_dims))
                index = []
                starts = [0] + out_dims[:-1]
                lengths = [out_dims[i] // out_dims[-1] for i in range(dilation_num)]
                for i in range(out_dims[-1]):
                    for j in range(dilation_num):
                        index += list(range(starts[j], starts[j] + lengths[j]))
                        starts[j] += lengths[j]
                self.index = index
                assert(len(index) == out_dim)
            self.out_dims = out_dims
        else:
            self.cat_out = False
            self.out_dims = [out_dim] * dilation_num

        if comb_mode in ('cat_in', 'cat_both'):
            if equal_dim:
                assert in_dim % dilation_num == 0
                in_dims = [in_dim // dilation_num] * dilation_num
            else:
                in_dims = [in_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                in_dims.append(in_dim - sum(in_dims))
            self.in_dims = in_dims
            self.cat_in = True
        else:
            self.cat_in = False
            self.in_dims = [in_dim] * dilation_num

        conv_type = DepthWiseSeperableConv if use_depthwise else nn.Conv2d
        dilation = min_dilation
        for i in range(dilation_num):
            if isinstance(padding, int):
                cur_padding = padding * dilation
            else:
                cur_padding = padding[i]
            convs.append(conv_type(
                self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs
            ))
            if i > 0 and shared_weights:
                convs[-1].weight = convs[0].weight
                convs[-1].bias = convs[0].bias
            dilation *= 2
        self.convs = nn.ModuleList(convs)

        self.shuffle_in_channels = shuffle_in_channels
        if self.shuffle_in_channels:
            # shuffle list as shuffling of tensors is nondeterministic
            in_channels_permute = list(range(in_dim))
            random.shuffle(in_channels_permute)
            # save as buffer so it is saved and loaded with checkpoint
            self.register_buffer('in_channels_permute', torch.tensor(in_channels_permute))

    def forward(self, x):
        if self.shuffle_in_channels:
            x = x[:, self.in_channels_permute]

        outs = []
        if self.cat_in:
            if self.equal_dim:
                x = x.chunk(len(self.convs), dim=1)
            else:
                new_x = []
                start = 0
                for dim in self.in_dims:
                    new_x.append(x[:, start:start+dim])
                    start += dim
                x = new_x
        for i, conv in enumerate(self.convs):
            if self.cat_in:
                input = x[i]
            else:
                input = x
            outs.append(conv(input))
        if self.cat_out:
            out = torch.cat(outs, dim=1)[:, self.index]
        else:
            out = sum(outs)
        return out



def get_conv_block_ctor(kind='default'):
    if not isinstance(kind, str):
        return kind
    if kind == 'default':
        return nn.Conv2d
    if kind == 'depthwise':
        return DepthWiseSeperableConv   
    if kind == 'multidilated':
        return MultidilatedConv
    raise ValueError(f'Unknown convolutional block kind {kind}')



def get_norm_layer(kind='bn'):
    if not isinstance(kind, str):
        return kind
    if kind == 'bn':
        return nn.BatchNorm2d
    if kind == 'in':
        return nn.InstanceNorm2d
    if kind == 'gn':
        return nn.GroupNorm
    raise ValueError(f'Unknown norm block kind {kind}')


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')



class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, conv_kind='default',
                 dilation=1, in_dim=None, groups=1, second_dilation=None, use_dino_cls=False):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        if second_dilation is None:
            second_dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout,
                                                conv_kind=conv_kind, dilation=dilation, in_dim=None, groups=groups,
                                                second_dilation=second_dilation)

        self.out_channnels = dim

        self.use_dino_cls = use_dino_cls
        if self.use_dino_cls:
            self.adapter_mlp = nn.Linear(768, 3*dim)
            # self.layer_norm = nn.LayerNorm(dim, 1e-6, elementwise_affine=False)


    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_kind='default',
                         dilation=1, in_dim=None, groups=1, second_dilation=1):
        conv_layer = get_conv_block_ctor(conv_kind)

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == 'zero':
            p = dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, dilation=dilation),
                       norm_layer(dim) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(second_dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(second_dilation)]
        elif padding_type == 'zero':
            p = second_dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, dilation=second_dilation, groups=groups),
                       norm_layer(dim) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, dim),]

        return nn.Sequential(*conv_block)

    def forward(self, x, dino_cls_token=None):
        if self.use_dino_cls:
            assert dino_cls_token is not None

            B,d = dino_cls_token.shape
            alpha,beta,gamma = self.adapter_mlp(dino_cls_token).reshape(B,3,self.dim).unsqueeze(-1).unsqueeze(-1).unbind(1)
            
    
        before_x = x

        if self.use_dino_cls:
            x_mean, x_var = calc_mean_std(x, eps=1e-6)
            x = (x-x_mean)/x_var # norm

            x = x*(1+alpha) + beta # scale and shift

        x = self.conv_block(x)

        if self.use_dino_cls:
            x = x * gamma # scale

        out = x + before_x
        return out



class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', conv_kind='default', activation=nn.ReLU(True),
                 use_skip=False,
                 up_norm_layer=nn.BatchNorm2d, affine=None,
                 up_activation=nn.ReLU(True), add_out_act=True,
                 max_features=1024, is_resblock_depthwise=False, 
                 dilation=1, second_dilation=None,
                 use_dino_cls=False,):
        assert (n_blocks >= 0)
        super().__init__()
        self.use_skip = use_skip
        print('use skip:', use_skip)

        self.use_dino_cls = use_dino_cls
        print('use dino class token feature:', use_dino_cls)

        conv_layer = get_conv_block_ctor(conv_kind)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)
        


        inp_module = [nn.ReflectionPad2d(3),
                 conv_layer(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, ngf),
                 activation]
        self.inp_module = nn.Sequential(*inp_module)

        identity = Identity()
        ### downsample
        self.downs = nn.ModuleList([])
        for i in range(n_downsampling):
            mult = 2 ** i

            self.downs.append(nn.ModuleList([
                conv_layer(min(max_features, ngf * mult),
                                min(max_features, ngf * mult * 2),
                                kernel_size=3, stride=2, padding=1),
                        norm_layer(min(max_features, ngf * mult * 2)) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, min(max_features, ngf * mult * 2)) ,
                        activation
            ]))

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)


        # resnet blocks
        self.mid_res = nn.ModuleList([])
        for i in range(n_blocks):
            if is_resblock_depthwise:
                resblock_groups = feats_num_bottleneck
            else:
                resblock_groups = 1

            self.mid_res.append(ResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_kind=conv_kind, groups=resblock_groups,
                                    dilation=dilation, second_dilation=second_dilation, use_dino_cls=use_dino_cls))

        # upsample
        self.ups = nn.ModuleList([])
        self.skip_linear = nn.ModuleList([])
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)

            if self.use_skip and i>0:
                inp_shape = min(max_features, ngf * mult) * 2
                out_shape = min(max_features, int(ngf * mult / 2))
            else:
                inp_shape = min(max_features, ngf * mult)
                out_shape = min(max_features, int(ngf * mult / 2))

            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(inp_shape, out_shape,
                                    kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))) if not issubclass(up_norm_layer, nn.GroupNorm) else up_norm_layer(16, min(max_features, int(ngf * mult / 2))), 
                      up_activation
            ]))
            
            if self.use_skip and i>0:
                self.skip_linear.append(nn.Linear(min(max_features, int(ngf * mult / 2))*2, min(max_features, int(ngf * mult / 2))))


        out_module = [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            out_module.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.out_module = nn.Sequential(*out_module)

    def forward(self, x, dino_cls_token=None):
        x = self.inp_module(x)

        res_fea = []
        for i, (conv, norm, act) in enumerate(self.downs):
            x = act(norm(conv(x)))
            
            if self.use_skip and i<(len(self.downs)-1):
                res_fea.append(x)
    
        
        for blk in self.mid_res:
            x = blk(x, dino_cls_token=dino_cls_token)

        for j, (conv, norm, act) in enumerate(self.ups):
            if self.use_skip and j>0:
                skip_x = res_fea.pop(-1)
                x = torch.cat([x, skip_x], dim=1)

            x = act(norm(conv(x)))
        
        x = self.out_module(x)

        return x



class ResNetEncoder(nn.Module):
    def __init__(self, gaussian_per_patch, hidden_dim, patch_size, encoder_args):
        super().__init__()

        patch_nc = hidden_dim*gaussian_per_patch

        self.encoder = GlobalGenerator(output_nc=hidden_dim, input_nc=3, **encoder_args)
        self.conv_out = nn.Conv2d(in_channels=hidden_dim, out_channels=patch_nc, kernel_size=patch_size, stride=patch_size)


    def forward(self, input, mask=None, dino_cls_token=None,):
        out = self.encoder(input, dino_cls_token=dino_cls_token) # B C H W
        out = self.conv_out(out) # B C H//p W//p
        B,C,_,_ = out.shape

        out = out.view(B,C,-1).permute(0,2,1)
        return out

