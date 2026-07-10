#!/bin/bash
#SBATCH --job-name=simgcd_cub
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=/userhome/cs/mkyahx/dev_outputs/submit_cub_%j.out
#SBATCH --error=/userhome/cs/mkyahx/dev_outputs/submit_cub_%j.err

set -e
set -x

mkdir -p /userhome/cs/mkyahx/dev_outputs
source /userhome/cs/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate simgcd
cd /userhome/cs/mkyahx/AFGCD/

CUDA_VISIBLE_DEVICES=0 python train.py \
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
    --exp_name cub_simgcd

conda deactivate
echo "Finish"
