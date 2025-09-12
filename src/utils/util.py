import os
import torch
import itertools
from torchvision.io import write_jpeg
from collections import defaultdict


root_dir = ""  # TODO
sample_dir = ""  # TODO
def get_log_name(data_type, dataname):
    save_dir = os.path.join(root_dir, data_type)
    os.makedirs(save_dir, exist_ok=True)

    num = len(os.listdir(save_dir)) + 1
    log_dir = save_dir + '/' + dataname + '_' + str(num) + '/'

    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def train_help(epoch, parameter, lr):
    parameter1, parameter2, parameter3, parameter4, parameter5, parameter6 = itertools.tee(parameter, 6)

    print('epoch:{}, lr:{}'.format(epoch, lr))

    vit_max_norm = max((p.grad.data.abs().max() if 'vit' in name else 0.) for name, p in parameter1)
    vit_average_norm = max((p.grad.data.abs().mean() if 'vit' in name else 0.) for name, p in parameter2)
    vit_parameter_norm = max((p.data.abs().mean() if 'vit' in name else 0.) for name, p in parameter3)
    print('vit: max norm is {}, average norm is {}, parameter norm:{}'.format(vit_max_norm, vit_average_norm, vit_parameter_norm))

    gs_max_norm = max((p.grad.data.abs().max() if 'vit' not in name else 0. ) for name, p in parameter4)
    gs_average_norm = max((p.grad.data.abs().mean() if 'vit' not in name else 0. ) for name, p in parameter5)
    gs_parameter_norm = max((p.data.abs().mean() if 'vit' not in name else 0. ) for name, p in parameter6)
    print('gaussian: max norm is {}, average norm is {}, parameter norm:{}'.format(gs_max_norm, gs_average_norm, gs_parameter_norm))
    
def get_sample_path(data_type, dataname, epoch_id):
    save_dir = os.path.join(sample_dir, data_type)
    os.makedirs(save_dir, exist_ok=True)

    log_dir = save_dir + '/' + dataname + '_' + str(epoch_id) + '/'
    os.makedirs(log_dir, exist_ok=True)

    return log_dir




def verify_img(imgs, mask=None, savedir=None):
    '''
    img: B 3 H W
    mask: B 1 H W
    '''
    bs = imgs.shape[0]

    for b in range(bs):
        img = imgs[b]
        img = (img*255).to(dtype=torch.uint8)
        p = os.path.join(savedir, f'img_{b}.png')
        write_jpeg(img.detach().cpu(), p)
        if mask is not None:
            mask_img = img * mask[b].to(dtype=torch.uint8)
            mask_p = os.path.join(savedir, f'mask_img_{b}.jpg')
            write_jpeg(mask_img.detach().cpu(), mask_p)

import random
import numpy as np
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze(*modules):
    """freeze the network for forward process"""
    for module in modules:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def unfreeze(*modules):
    """ unfreeze the network for parameter update"""
    for module in modules:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def to_torch_dtype(dtype):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp64':
        return torch.float64
    else:
        return None
    

def format_numel(numel):
    units = ['', 'K', 'M', 'B']
    unit_index = 0
    while numel >= 1000 and unit_index < len(units) - 1:
        numel /= 1000.0
        unit_index += 1

    if unit_index > 0:
        if numel >= 10:
            return f"{numel:.1f}{units[unit_index]}"
        else:
            return f"{numel:.2f}{units[unit_index]}"
    else:
        return f"{int(numel)}"