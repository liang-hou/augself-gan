#!/bin/bash
python train.py --experiment_name augself+biggan-cifar100 --DiffAugment color,translation,cutout \
--augself color,translation,cutout --translation 0.25 --cutout 0.75 --D_augself 1.0 --G_augself 1.0 \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 2000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 4000 --seed 0