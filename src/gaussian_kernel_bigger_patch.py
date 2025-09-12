import torch
from torch import Tensor
from torch.autograd import Function

from jaxtyping import Float
from typing import Tuple

import gspaint_cuda

def project_bigger_patch_gs(
    mean: Float[Tensor, "gsbank 2"],
    L_covariance: Float[Tensor, "gsbank 3"],
    patch_size: Tuple[int, int],
    patch_num: Tuple[int, int],
):
    return project_bigger_patch_gs_function.apply(
        mean.contiguous(),
        L_covariance.contiguous(),
        patch_size,
        patch_num
    )


def rasterize_bigger_patch_gs(
    xys: Float[Tensor, "patch gsbank 2"],
    conics: Float[Tensor, "patch gsbank 3"],
    colors: Float[Tensor, "patch gsbank 3"],
    coefficient: Float[Tensor, "patch gsbank 1"],
    image_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    patch_num:  Tuple[int, int],
):    

    return rasterize_bigger_patch_gs_function.apply(
        xys.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        coefficient.contiguous(),
        image_size,
        patch_size,
        patch_num,
    )



class project_bigger_patch_gs_function(Function):
    @staticmethod
    def forward(ctx, mean, L_covariance, patch_size, patch_num):
        gsbank_num = mean.shape[1]

        (
            xy,
            conics,
        )=gspaint_cuda.project_gs_inpaint_bigger_patch_forward(
            mean, 
            L_covariance,
            gsbank_num,
            patch_size,
            patch_num
        )

        ctx.save_for_backward(
            mean,
            L_covariance,
            conics
        )
        ctx.gsbank_num = gsbank_num
        ctx.patch_size = patch_size
        ctx.patch_num = patch_num

        xy = xy.reshape(-1, gsbank_num, 2)
        conics = conics.reshape(-1, gsbank_num, 3)

        return (xy, conics)
    

    @staticmethod
    def backward(ctx, v_xys, v_conics):
        (
            mean,
            L_covariance,
            conics,
        ) = ctx.saved_tensors 


        (v_means, v_L_elements) = gspaint_cuda.project_gs_inpaint_bigger_patch_backward(
            ctx.gsbank_num,
            ctx.patch_size,
            ctx.patch_num, 
            mean,
            L_covariance,
            conics,
            v_xys,
            v_conics,
        )

        v_means = v_means.reshape(-1, ctx.gsbank_num, 2)
        v_L_elements = v_L_elements.reshape(-1, ctx.gsbank_num, 3)

        return(
            v_means,       # gradient for input: means
            v_L_elements,  # gradient for input: L_elements
            None,          # no gradient for input:patch_size
            None,          # no gradient for input:patch num
        )


class rasterize_bigger_patch_gs_function(Function):

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "gsbank 2"],
        conics: Float[Tensor, "gsbank 3"],
        colors: Float[Tensor, "gsbank 3"],
        coefficient: Float[Tensor, "patchnum gsbank"],
        image_size:Tuple[int, int],
        patch_size:Tuple[int, int],
        patch_num:Tuple[int, int], 
    ):
        gsbank_num = xys.shape[1]

        out_img = gspaint_cuda.rasterize_gs_inpaint_bigger_patch_forward(
            gsbank_num,
            patch_size,
            image_size,
            patch_num,
            xys,
            conics,
            colors,
            coefficient,
        )

        ctx.save_for_backward(
            xys, conics, colors, coefficient
        )

        ctx.image_size = image_size
        ctx.patch_size = patch_size
        ctx.patch_num = patch_num
        ctx.gsbank_num = gsbank_num

        return out_img
    

    @staticmethod
    def backward(ctx, v_out_img):

        (
            xys,
            conics,
            colors,
            coefficient,
        ) = ctx.saved_tensors
        gsbank_num = ctx.gsbank_num

        v_xys, v_conics, v_colors, v_coefficient = gspaint_cuda.rasterize_gs_inpaint_bigger_patch_backward(
                ctx.image_size,
                ctx.patch_size,
                ctx.patch_num,
                xys,
                conics,
                colors,
                coefficient,
                v_out_img,
            )
        
        v_xys = v_xys.reshape(-1, gsbank_num, 2)
        v_conics = v_conics.reshape(-1, gsbank_num, 3)
        v_colors = v_colors.reshape(-1, gsbank_num, 3)
        v_coefficient = v_coefficient.reshape(-1, gsbank_num, 1)



        return (
            v_xys,           # gradient for input: xys, used to pass to previous layer
            v_conics,        # gradient for input: conics,  used to pass to previous layer
            v_colors,        # gradient for input: colors,  used to update the parameter
            None,            # no gradient for input: coefficient
            None,            # no gradient for input: image_size
            None,            # no gradient for input: patch_size
            None,            # no gradient for input: patch_num
        )