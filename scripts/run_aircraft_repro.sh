#!/bin/bash

set -e
set -x

for seed in 0; do
    CUDA_VISIBLE_DEVICES=0 python train_repro.py \
        --dataset_name 'aircraft' \
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
        --seed $seed \
        --exp_name aircraft_simgcd_seed_${seed}
done
