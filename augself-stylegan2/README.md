# AugSelf-StyleGAN2

This repo is implemented upon [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with minimal modifications to train and load augself-stylegan2 models in PyTorch. Please check the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README for the dependencies and the other usages of this codebase. To train on the FFHQ and LSUN-Cat datasets, please follow the guidelines in the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README to prepare the datasets.

## FFHQ and LSUN-Cat

The following command are the examples of training AugSelf-StyleGAN2 with the default *Color + Translation + Cutout* AugSelf on FFHQ and LSUN-Cat with different training samples with 1 GPU.
```bash
python train.py --outdir=training-runs --gpus=1 --data=/path/to/ffhq --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=1000
python train.py --outdir=training-runs --gpus=1 --data=/path/to/ffhq --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=5000
python train.py --outdir=training-runs --gpus=1 --data=/path/to/ffhq --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=10000
python train.py --outdir=training-runs --gpus=1 --data=/path/to/ffhq --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=30000

python train.py --outdir=training-runs --gpus=1 --data=/path/to/lsuncat --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=1000
python train.py --outdir=training-runs --gpus=1 --data=/path/to/lsuncat --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=5000
python train.py --outdir=training-runs --gpus=1 --data=/path/to/lsuncat --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=10000
python train.py --outdir=training-runs --gpus=1 --data=/path/to/lsuncat --mirror=true --cfg=auto --augself color,translation,cutout --d_augself 1 --g_augself 0.2 --subset=30000
```

## Low-Shot Generation

The following commands are the examples of training AugSelf-StyleGAN2 with the *Color* AugSelf on five low-shot datasets with 1 GPU.
```bash
python train.py --outdir=training-runs --gpus=1 --kimg=5000 --augself color --d_augself 1 --g_augself 1 --data=https://data-efficient-gans.mit.edu/datasets/100-shot-obama.zip
python train.py --outdir=training-runs --gpus=1 --kimg=5000 --augself color --d_augself 0.1 --g_augself 0.1 --data=https://data-efficient-gans.mit.edu/datasets/100-shot-grumpy_cat.zip
python train.py --outdir=training-runs --gpus=1 --kimg=5000 --augself color --d_augself 1 --g_augself 1 --data=https://data-efficient-gans.mit.edu/datasets/100-shot-panda.zip
python train.py --outdir=training-runs --gpus=1 --kimg=5000 --augself color --d_augself 0.1 --g_augself 0.1 --data=https://data-efficient-gans.mit.edu/datasets/AnimalFace-cat.zip
python train.py --outdir=training-runs --gpus=1 --kimg=5000 --augself color --d_augself 1 --g_augself 1 --data=https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip
```
