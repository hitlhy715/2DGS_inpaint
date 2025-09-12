import torch
import torch.nn as nn
from itertools import product
import numpy as np
import math

from src.utils.block import MLP
from .base import map_base


def get_uniform_points(num_points):

    x_coords = np.linspace(-1, 1, int(np.sqrt(num_points)))
    y_coords = np.linspace(-1, 1, int(np.sqrt(num_points)))
    

    xv, yv = np.meshgrid(x_coords, y_coords)
    

    points = np.vstack((xv.flatten(), yv.flatten())).T
    

    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        additional_points = points[np.random.choice(len(points), num_points - len(points))]
        points = np.vstack((points, additional_points))
    
    return points


def get_coord(width, height):
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)


    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')


    x_grid = 2 * (x_grid / (width)) - 1 #+ 1/width
    y_grid = 2 * (y_grid / (height)) - 1 #+ 1/height


    coordinates = torch.stack((y_grid, x_grid), dim=-1).reshape(-1, 2)
    
    return coordinates


def is_perfect_square(num):

    if num < 0:
        return False

    root = math.sqrt(num)

    return int(root) ** 2 == num



class DirectMap(nn.Module, map_base):
    def __init__(self, hidden_dim, gaussian_per_patch, out_patch_size, out_image_size, image_size, window_size=1, gs_fix=False):
        super().__init__()

        self.color = None
        self.cholesky = None
        self.offset = None

        self.window_size = window_size
        self.gaussian_per_patch = gaussian_per_patch
        self.hidden_dim = hidden_dim
        self.gs_fix = gs_fix
        self.out_patch_size = out_patch_size
        self.out_image_size = out_image_size
        self.image_size = image_size

        assert is_perfect_square(gaussian_per_patch), 'must be square number'
        self.sqrt_gaussian = int(math.sqrt(gaussian_per_patch))

        # cholesky related
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]).cuda()
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]).cuda()
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]).cuda()
        
        self.gau_dict = torch.tensor(list(product(cho1, cho2, cho3))).cuda()
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1,3).cuda()), dim=0) # shape:[344,3]

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.cholesky_linear = nn.Linear(hidden_dim, hidden_dim)
        # self.cholesky_linear = nn.Linear(hidden_dim, 730)
        self.mlp_gaudict = MLP(
            in_dim=3,
            out_dim=hidden_dim,
            hidden_list=[hidden_dim//2, hidden_dim, hidden_dim, hidden_dim]
        )

        # color related
        self.color_mlp = MLP(
            in_dim=hidden_dim,
            out_dim=3,
            hidden_list= [hidden_dim,hidden_dim,hidden_dim,hidden_dim//2]
        )

        
        # xy related
        self.mlp_xy = MLP(
            in_dim=hidden_dim,
            out_dim=2,
            hidden_list=[hidden_dim,hidden_dim,hidden_dim//2]
        )

    def map(self, feat, bs):
        # get_color
        color_feat = feat
        color_feat = color_feat.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)

        self.color = self.color_mlp(color_feat)

        # get cholesky
        cholesky_fea = feat
        cholesky_fea = cholesky_fea.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)
        cholesky_fea = self.leaky_relu(cholesky_fea)
        cholesky_fea = self.cholesky_linear(cholesky_fea)

        vector = self.mlp_gaudict(self.gau_dict.to(cholesky_fea.device))
        vector = vector.permute(1,0)
        cholesky_weight = cholesky_fea @ vector
        cholesky_weight = torch.softmax(cholesky_weight, dim=-1)
        self.cholesky = cholesky_weight @ self.gau_dict.to(cholesky_weight.device) # B 256 20 3


        # get_xy
        xy_feat = feat
        xy_feat = xy_feat.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)
        offset = self.mlp_xy(xy_feat)
        self.offset = torch.tanh(offset) # B 256 30 2
    
    def get_iter(self, i):
        offset_ = self.offset[i,:,:,:]
        offset_ = offset_.squeeze(0) # 256,36,2
        color_ = self.color[i, :, :]
        color_ = color_.squeeze(0) # 256,36,3
        para_ = self.cholesky[i, :, :]
        para_ = para_.squeeze(0) # 256,36,3

        get_xyz = torch.tensor(get_coord(self.sqrt_gaussian, self.sqrt_gaussian)).reshape(self.sqrt_gaussian, self.sqrt_gaussian, 2).cuda() 
        get_xyz = get_xyz.reshape(-1,2) # 36,2
        
        if self.gs_fix:
            patch_n = offset_.shape[0]
            get_xyz = get_xyz.unsqueeze(0).repeat(patch_n,1,1)
        else:   
            xyz1 = get_xyz[:,0:1] + 2*self.window_size*offset_[:,:,0:1]/self.out_patch_size[1] - 1/self.out_patch_size[1]    # -  1/lr_w
            xyz2 = get_xyz[:,1:2] + 2*self.window_size*offset_[:,:,1:2]/self.out_patch_size[0] - 1/self.out_patch_size[0]    # -  1/lr_h
            get_xyz = torch.cat((xyz1, xyz2), dim = -1) # 256,36,2

        
        # weighted_cholesky = para_/4
        # weighted_cholesky  =para_
        weighted_cholesky = para_*(self.out_image_size[0]/self.image_size[0])

        opacity = torch.ones(color_.shape[0], color_.shape[1], 1).cuda()

        return get_xyz, weighted_cholesky, color_, opacity
