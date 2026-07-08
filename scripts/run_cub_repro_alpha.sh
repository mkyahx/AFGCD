#!/bin/bash

set -e
set -x

MASK_ROOT=${MASK_ROOT:-/userhome/cs/mkyahx/AFGCD/masks}
ALPHA=${ALPHA:-0.7}

for seed in 1; do
    CUDA_VISIBLE_DEVICES=0 python train_repro_alpha.py \
        --dataset_name 'cub' \
        --batch_size 128 \
        --grad_from_block 11 \
        --epochs 200 \
        --num_workers 8 \
        --use_ssb_splits \
        --sup_weight 0.35 \
        --weight_decay 5e-5 \
        --transform 'imagenet' \
        --lr 0.1 \
        --eval_funcs 'v2' \
        --warmup_teacher_temp 0.07 \
        --teacher_temp 0.04 \
        --warmup_teacher_temp_epochs 30 \
        --memax_weight 2 \
        --threshold 0.2 \
        --mask_root "$MASK_ROOT" \
        --alpha "$ALPHA" \
        --seed $seed \
        --exp_name cub_simgcd_alpha_${ALPHA}_seed_${seed}
done
