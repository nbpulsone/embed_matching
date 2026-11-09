#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -J "ZEROSHOT_EMBED_MATCH_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100|A100-80G|A30|V100|P100|L40S"
#SBATCH -x gpu-6-[01-20]

#SBATCH -o ./logs/embed_match_bert_zeroshot_%j.out
#SBATCH -e ./logs/embed_match_bert_zeroshot_%j.err

# interactive command: srun -N 1 -n 2 --mem=32768 --time=120 --partition=short --gres=gpu:1 -x "gpu-6-[01-20]" --pty /usr/bin/bash
module load python/3.11.12/7vjddo3
source /home/nbpulsone/sandbox_research/rodoem_llm_ft/modenv/bin/activate

#temperatures=(0.05 0.07 0.1)
#for temp in "${temperatures[@]}"; do
    #echo "==================="
    #echo "TESTING TEMPERATURE: $temp"
    #echo "===================" 

    #python train_ditto.py \
        #--task Structured/Beer \
        #--lm_left bert \
        #--lm_right roberta \
        #--projection_dim 256 \
        #--batch_size 64 \
        #--max_len 128 \
        #--n_epochs 40 \
        #--lr 3e-5 \
        #--loss infonce \
        #--temperature "$temp" \
        #--bf16 \
        #--save_model
    #done

python train_ditto.py \
    --task Structured/Beer \
    --lm distilbert \
    --zero_shot \
    --zero_shot_threshold 0.5 \
    --bf16

python train_ditto.py \
    --task Structured/Beer \
    --lm_left bert \
    --lm_right roberta \
    --zero_shot \
    --zero_shot_threshold 0.5 \
    --bf16
