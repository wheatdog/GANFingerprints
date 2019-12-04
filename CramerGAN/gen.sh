#!/bin/bash

cnt=0
num_per_round=10000
for i in {0..19}; do
    CUDA_VISIBLE_DEVICES=6 \
        python gan/main.py \
        --dataset celebA \
        --data_dir ~/datasets/img_align_celeba_png_crop_128x128/ \
        --checkpoint_dir models/ \
        --output_dir_of_test_samples gen/celeba_align_png_cropped/ \
        --no_of_samples $num_per_round \
        --start_idx $cnt \
        --model cramer --name cramer_gan \
        --architecture g_resnet5 --output_size 128 --dof_dim 256 \
        --gradient_penalty 10. \
        --MMD_lr_scheduler \
        --random_seed $i
    cnt=$((cnt+num_per_round))
done
