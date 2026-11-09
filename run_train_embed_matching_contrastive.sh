#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -J "HETEROGENOUS_CONTRASTIVE_EMBED_MATCH_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100|A100-80G|A30|V100|P100|L40S"
#SBATCH -x gpu-6-[01-20]

#SBATCH -o ./logs/embed_match_hetero_contrastive_%j.out
#SBATCH -e ./logs/embed_match_hetero_contrastive_%j.err

# interactive command: srun -N 1 -n 2 --mem=32768 --time=120 --partition=short --gres=gpu:1 -x "gpu-6-[01-20]" --pty /usr/bin/bash
module load python/3.11.12/7vjddo3
source /home/nbpulsone/sandbox_research/rodoem_llm_ft/modenv/bin/activate

margins=(0.0 0.3 0.5)
for margin in "${margins[@]}"; do
    echo "==================="
    echo "TESTING MARGIN: $margin"
    echo "===================" 

    python train_ditto.py \
        --task Structured/Beer \
        --lm_left bert \
        --lm_right roberta \
        --projection_dim 256 \
        --batch_size 64 \
        --max_len 128 \
        --n_epochs 40 \
        --lr 3e-5 \
        --loss contrastive \
        --contrastive_margin "$margin" \
        --bf16 \
        --save_model
    done
