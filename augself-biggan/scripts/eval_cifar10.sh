#!/bin/bash
python eval.py --experiment_name augself-biggan-cifar10 --DiffAugment color,translation,cutout \
--SS_augs color,translation,cutout --SS_arch linear --SS_fuse sub --SS_label sym --SS_margin 0.0 --SS_D_data all --SS_G_loss both --SS_D_weight 1.0 --SS_G_weight 1.0 \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 2000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 4000 --seed 0 \
--network=weights/augself-biggan-cifar10-0.1/G_ema_best.pth
