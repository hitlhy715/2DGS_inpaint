import torch

def overlap_rasterize(img, patch_size, patch_num, overlap_pad=1):
    if overlap_pad == 0:
        return img
    assert patch_size[0] * patch_num[0] == img.shape[0], f"Height mismatch: {patch_size[0]}*{patch_num[0]} != {img.shape[0]}"
    assert patch_size[1] * patch_num[1] == img.shape[1], f"Width mismatch: {patch_size[1]}*{patch_num[1]} != {img.shape[1]}"
    assert patch_size[0] > 4 * overlap_pad, f"patch_size[0] must be > 4*overlap_pad ({4*overlap_pad}), got {patch_size[0]}"
    assert patch_size[1] > 4 * overlap_pad, f"patch_size[1] must be > 4*overlap_pad ({4*overlap_pad}), got {patch_size[1]}"

    img = img.permute(2, 0, 1)
    C, H, W = img.shape
    patch_H, patch_W = patch_num
    patch_size_H, patch_size_W = patch_size

    target_patch_size_H = patch_size_H - 2 * overlap_pad
    target_patch_size_W = patch_size_W - 2 * overlap_pad

    img_patch = img.unfold(1, patch_size_H, patch_size_H).unfold(2, patch_size_W, patch_size_W) 


    current_upper = img_patch[..., overlap_pad:2*overlap_pad, overlap_pad:-overlap_pad] 
    upper_prev = img_patch[:, :-1, :, -overlap_pad:, overlap_pad:-overlap_pad] if patch_H > 1 else None  
    if patch_H > 1:
        upper_combined = (current_upper[:, 1:, :, :] + upper_prev) / 2
        upper_edge = torch.cat([current_upper[:, :1, :, :], upper_combined], dim=1)
    else:
        upper_edge = current_upper
    
    current_lower = img_patch[..., -2*overlap_pad:-overlap_pad, overlap_pad:-overlap_pad] 
    lower_next = img_patch[:, 1:, :, :overlap_pad, overlap_pad:-overlap_pad] if patch_H > 1 else None 
    if patch_H > 1:
        lower_combined = (current_lower[:, :-1, :, :] + lower_next) / 2
        lower_edge = torch.cat([lower_combined, current_lower[:, -1:, :, :]], dim=1)
    else:
        lower_edge = current_lower
    

    current_left = img_patch[..., overlap_pad:-overlap_pad, overlap_pad:2*overlap_pad]
    left_prev = img_patch[:, :, :-1, overlap_pad:-overlap_pad, -overlap_pad:] if patch_W > 1 else None
    if patch_W > 1:
        left_combined = (current_left[:, :, 1:, :] + left_prev) / 2
        left_edge = torch.cat([current_left[:, :, :1, :], left_combined], dim=2)
    else:
        left_edge = current_left
    
    current_right = img_patch[..., overlap_pad:-overlap_pad, -2*overlap_pad:-overlap_pad]
    right_next = img_patch[:, :, 1:, overlap_pad:-overlap_pad, :overlap_pad] if patch_W > 1 else None
    if patch_W > 1:
        right_combined = (current_right[:, :, :-1, :] + right_next) / 2
        right_edge = torch.cat([right_combined, current_right[:, :, -1:, :]], dim=2)
    else:
        right_edge = current_right
    

    inner_part = img_patch[..., 2*overlap_pad:-2*overlap_pad, 2*overlap_pad:-2*overlap_pad]
    
    out_patch = torch.zeros((C, patch_H, patch_W, target_patch_size_H, target_patch_size_W), 
                            dtype=img.dtype, device=img.device)
    

    out_patch[..., :overlap_pad, :] = upper_edge
    out_patch[..., -overlap_pad:, :] = lower_edge
    out_patch[..., :, :overlap_pad] = left_edge
    out_patch[..., :, -overlap_pad:] = right_edge
    if target_patch_size_H > 2 and target_patch_size_W > 2:
        out_patch[..., overlap_pad:-overlap_pad, overlap_pad:-overlap_pad] = inner_part
    
    out_img = out_patch.permute(0, 1, 3, 2, 4).contiguous()
    out_img = out_img.view(C, patch_H * target_patch_size_H, patch_W * target_patch_size_W)
    out_img = out_img.permute(1, 2, 0)
    
    torch.cuda.empty_cache()
    return out_img
