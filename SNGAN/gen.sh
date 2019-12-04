#!/bin/bash

CUDA_VISIBLE_DEVICES=5 \
    python evaluations/gen_images.py \
    --config_path configs/sn_projection_celeba.yml \
    --snapshot models/celeba_align_png_cropped.npz \
    --results_dir gen/celeba_align_png_cropped \
    --num_pngs 200000 \
    --seed 0
