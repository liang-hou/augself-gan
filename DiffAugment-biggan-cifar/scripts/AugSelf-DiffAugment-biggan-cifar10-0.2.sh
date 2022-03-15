#!/bin/bash
python train.py --experiment_name AugSelf-DiffAugment-biggan-cifar10-0.2 --DiffAugment color,translation,cutout \
--augself color,translation,cutout,rotation --D_augself_lambda 10.0 --G_augself_lambda 10.0 \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 --num_samples 10000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 4000 --seed 0