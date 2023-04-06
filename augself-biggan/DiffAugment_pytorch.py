# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import numpy as np
import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True, config={}):
    param = {}
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            ys = []
            for f in AUGMENT_FNS[p]:
                x, y = f(x, config)
                ys.append(y)
            ys = torch.cat(ys, 1)
            param[p] = ys
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x, param


def rand_brightness(x, config):
    ratio = config.get('SS_brightness', 1.0)
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x = x + (r - 0.5) * ratio
    return x, r.view(-1, 1)


def rand_saturation(x, config):
    ratio = config.get('SS_saturation', 1.0)
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * ((r * 2 - 1) * ratio + 1) + x_mean
    return x, r.view(-1, 1)


def rand_contrast(x, config):
    ratio = config.get('SS_contrast', 1.0)
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * ((r - 0.5) * ratio + 1) + x_mean
    return x, r.view(-1, 1)


def rand_translation(x, config):
    ratio = config.get('SS_translation', 0.125)
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x, torch.cat([translation_x.view(-1, 1).float() / shift_x / 2.0 + 0.5, translation_y.view(-1, 1).float() / shift_y / 2.0 + 0.5], 1)


def rand_cutout(x, config):
    ratio = config.get('SS_cutout', 0.5)
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, torch.cat([offset_x.view(-1, 1).float() / (x.size(2) - cutout_size[0] % 2), offset_y.view(-1, 1).float() / (x.size(3) - cutout_size[1] % 2)], 1)


def rand_cutout_test(x, low_ratio=0.25, high_ratio=0.5):
    cutout_sizes = int(x.size(2) * high_ratio + 0.5), int(x.size(3) * high_ratio + 0.5)
    cutout_size_x = torch.randint(int(x.size(2) * low_ratio + 0.5), int(x.size(2) * high_ratio + 0.5) + 1, size=[x.size(0), 1, 1], device=x.device)
    cutout_size_y = torch.randint(int(x.size(3) * low_ratio + 0.5), int(x.size(3) * high_ratio + 0.5) + 1, size=[x.size(0), 1, 1], device=x.device)
    x_size_2 = torch.where(cutout_size_x % 2 == 0, torch.tensor(x.size(2) + 1, device=x.device), torch.tensor(x.size(2), device=x.device))
    x_size_3 = torch.where(cutout_size_y % 2 == 0, torch.tensor(x.size(3) + 1, device=x.device), torch.tensor(x.size(3), device=x.device))
    offset_x = (torch.rand(size=[x.size(0), 1, 1], device=x.device) * x_size_2).long()
    offset_y = (torch.rand(size=[x.size(0), 1, 1], device=x.device) * x_size_3).long()

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange((-cutout_sizes[0]+1) // 2, (cutout_sizes[0]+1) // 2, dtype=torch.long, device=x.device),
        torch.arange((-cutout_sizes[1]+1) // 2, (cutout_sizes[0]+1) // 2, dtype=torch.long, device=x.device),
    )

    # grid_x = torch.clamp(grid_x, min=-(cutout_size_x // 2), max=(cutout_size_x - 1) // 2)
    # grid_y = torch.clamp(grid_y, min=-(cutout_size_y // 2), max=(cutout_size_y - 1) // 2)
    grid_x_np = grid_x.cpu().numpy()
    grid_y_np = grid_y.cpu().numpy()
    cutout_size_x_np = cutout_size_x.cpu().numpy()
    cutout_size_y_np = cutout_size_y.cpu().numpy()
    grid_x_np_clip = np.clip(grid_x_np, a_min=-(cutout_size_x_np // 2), a_max=(cutout_size_x_np - 1) // 2)
    grid_y_np_clip = np.clip(grid_y_np, a_min=-(cutout_size_y_np // 2), a_max=(cutout_size_y_np - 1) // 2)
    grid_x = torch.from_numpy(grid_x_np_clip).to(x.device)
    grid_y = torch.from_numpy(grid_y_np_clip).to(x.device)
    grid_x = torch.clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, torch.cat([
        offset_x.view(-1, 1).float() / (x_size_2.view(-1, 1) - 1), 
        offset_y.view(-1, 1).float() / (x_size_3.view(-1, 1) - 1),
        (cutout_size_x.view(-1, 1).float() - int(x.size(2) * low_ratio + 0.5)) / (int(x.size(2) * high_ratio + 0.5) - int(x.size(2) * low_ratio + 0.5)),
        (cutout_size_y.view(-1, 1).float() - int(x.size(3) * low_ratio + 0.5)) / (int(x.size(3) * high_ratio + 0.5) - int(x.size(3) * low_ratio + 0.5))], 1)


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout]
}

AUGMENT_DIM = {
    'color': 3,
    'translation': 2,
    'cutout': 2
}