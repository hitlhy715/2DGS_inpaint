# discrminator architecture from https://github.com/CompVis/stable-diffusion
import functools
import torch
import torch.nn as nn


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """
    device = discriminator.device
    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1)).to(device=device)

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



from torch.nn.utils import spectral_norm
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_type='gn', use_dino_cls=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if norm_type == 'act': 
            norm_layer = ActNorm
        elif norm_type == 'gn':
            norm_layer = nn.GroupNorm
        elif norm_type == 'bn':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'sn':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        
        self.use_dino_cls = use_dino_cls
        self.adaln_mlp =  nn.ModuleList([])

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv1 = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias) if norm_layer!='sn'\
                                        else spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            sequence += [
                conv1,
                norm_layer(ndf * nf_mult) if norm_layer != nn.GroupNorm else norm_layer(16, ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

            if self.use_dino_cls:
                self.adaln_mlp.append(
                    nn.Linear(768, ndf * nf_mult*2)
                )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv2 = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias) if norm_layer!='sn'\
                    else spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        sequence += [
            conv2,
            norm_layer(ndf * nf_mult) if norm_layer != nn.GroupNorm else norm_layer(16, ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]
        if self.use_dino_cls:
            self.adaln_mlp.append(
                nn.Linear(768, ndf * nf_mult*2)
            )


        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        if self.use_dino_cls:
            self.init_adaln()

    
    def init_adaln(self):
        for i in range(len(self.adaln_mlp)):
            # alpha: init 0
            # beta: init 0
            # gamma: init 1

            out_channel = self.adaln_mlp[i].out_features
            half_out_channel = out_channel//2

            self.adaln_mlp[i].weight.data[:half_out_channel] = 0.0
            self.adaln_mlp[i].bias.data[:half_out_channel] = 0.0

            self.adaln_mlp[i].weight.data[half_out_channel:] = 0.0
            self.adaln_mlp[i].bias.data[half_out_channel:] = 1.0



    def forward(self, x, dino_fea=None):
        """Standard forward."""

        if self.use_dino_cls:
            assert dino_fea is not None
            dino_mlp_feas = []
            for i in range(len(self.adaln_mlp)):
                f = self.adaln_mlp[i](dino_fea)
                c = f.shape[-1]
                c1 = int(c*0.25)
                c2 = int(c*0.25)
                c3 = c-c1-c2

                alpha, beta, gamma = torch.split(f, [c1,c2,c3], dim=-1) 
                dino_mlp_feas.append([alpha, beta, gamma])



        blk_num = 0
        for name,module in self.main.named_children():
            # 2,3,4; 
            # 5,6,7; 
            # 8,9,10;
            if self.use_dino_cls and (name == '2' or name == '5' or name == '8'):
                idx = (int(name)-2)//3
                alpha, beta, _ = dino_mlp_feas[idx]

                x_mean, x_var = calc_mean_std(x, eps=1e-6)
                alpha = alpha.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)

                x = (x-x_mean)/x_var

                x = x*(1+alpha)+beta

            x = module(x)

            if self.use_dino_cls and (name == '4' or name == '7' or name == '10'):
                idx = (int(name)-4)//3
                _,_,gamma = dino_mlp_feas[idx]
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)

                x = x*gamma

                blk_num = blk_num + 1

        return x



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    def format_num(num):
        for unit in ['', 'K', 'M', 'G', 'T']:
            if num < 1000.0:
                return f"{num:.2f}{unit}"
            num /= 1000.0
        return f"{num:.2f}P"
    
    total_params_formatted = format_num(total_params)
    trainable_params_formatted = format_num(trainable_params)
    
    return total_params, trainable_params, total_params_formatted, trainable_params_formatted


if __name__ == '__main__':
    d = NLayerDiscriminator(use_dino_cls=True).apply(weights_init)
    input = torch.rand(3,3,256,256)
    dino = torch.rand(3,768)
    out = d(input, dino)
    print(out.shape)

    total_params, trainable_params, total_params_formatted, trainable_params_formatted = count_parameters(d)
    print(f" {total_params_formatted}")
    print(f" {trainable_params_formatted}")

