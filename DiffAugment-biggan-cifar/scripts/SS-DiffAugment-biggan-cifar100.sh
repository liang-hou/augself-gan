#!/bin/bash
python train.py --experiment_name ss-color+translation+cutout-linear-sub-d1g1-DiffAugment-biggan-cifar100 --DiffAugment color,translation,cutout \
--SS_augs color,translation,cutout --SS_arch linear --SS_fuse sub --SS_label same --SS_margin 0.0 --SS_D_data real --SS_G_loss ns --SS_D_weight 1.0 --SS_G_weight 1.0 \
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