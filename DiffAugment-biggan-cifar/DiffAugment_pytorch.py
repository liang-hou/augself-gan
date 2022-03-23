# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True):
    augself = {}
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            ys = []
            for f in AUGMENT_FNS[p]:
                x, y = f(x)
                ys.append(y)
            ys = torch.cat(ys, 1)
            augself[p] = ys
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x, augself


def rand_brightness(x):
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x = x + (r - 0.5)
    return x, r.view(-1, 1)


def rand_saturation(x):
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (r * 2) + x_mean
    return x, r.view(-1, 1)


def rand_contrast(x):
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (r + 0.5) + x_mean
    return x, r.view(-1, 1)


def rand_translation(x, ratio=0.125):
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
    return x, torch.cat([translation_x.view(-1, 1).float() / shift_x, translation_y.view(-1, 1).float() / shift_y], 1)


def rand_cutout(x, ratio=0.5):
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


def rand_rotation(x):
    degree = torch.randint(0, 4, size=[x.size(0),1,1,1], device=x.device)
    x = torch.where(degree == 1, x.transpose(2, 3).flip(3), x)
    x = torch.where(degree == 2, x.flip(2).flip(3), x)
    x = torch.where(degree == 3, x.transpose(2, 3).flip(2), x)
    return x, degree.view(-1, 1)


def rand_permutation(x):
    permutation = torch.randint(0, 6, size=[x.size(0),1,1,1], device=x.device)
    r, g, b = torch.split(x, 1, 1)
    x = torch.where(permutation == 1, torch.cat([r, b, g], 1), x)
    x = torch.where(permutation == 2, torch.cat([g, r, b], 1), x)
    x = torch.where(permutation == 3, torch.cat([g, b, r], 1), x)
    x = torch.where(permutation == 4, torch.cat([b, r, g], 1), x)
    x = torch.where(permutation == 5, torch.cat([b, g, r], 1), x)
    return x, permutation.view(-1, 1)


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'rotation': [rand_rotation],
    'permutation': [rand_permutation]
}

AUGMENT_DMS = {
    'color': 3,
    'translation': 2,
    'cutout': 2,
    'rotaion': 4,
    'permutation': 6
}

AUGMENT_TPS = {
    'color': 'regression',
    'translation': 'regression',
    'cutout': 'regression',
    'rotation': 'classification',
    'permutation': 'classification',
}