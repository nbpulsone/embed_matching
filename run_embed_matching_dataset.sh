#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100|A100-80G|A30|V100|P100|L40S"
#SBATCH -x gpu-6-[01-20]

# One job runs six methods sequentially for the dataset given as $1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /etc/profile
module load python/3.11.12/7vjddo3
source /home/nbpulsone/sandbox_research/rodoem_llm_ft/modenv/bin/activate

TASK="${1:?Usage: sbatch run_embed_matching_dataset.sh 'Structured/Beer'}"
case "$TASK" in
  Structured/Beer|Textual/Abt-Buy|Dirty/iTunes-Amazon) ;;
  *) echo "Unexpected task: $TASK" >&2; exit 2 ;;
esac
TAG="${TASK//\//_}"
OUTDIR="results/embed_matching_hetero_noproj/${TAG}"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results.csv"
printf 'dataset,method,left_encoder,right_encoder,zero_shot,margin,threshold,val_f1,test_f1\n' > "$CSV"

# Fixed controls, matching the supplied script's training parameters.
BATCH_SIZE=64
MAX_LEN=128
EPOCHS=40
LR=3e-5
MARGIN=0.5
ZERO_SHOT_THRESHOLD=0.5
PROJ_DIM=0

run_method() {
    local method="$1" left="$2" right="$3" mode="$4"
    local logfile="$OUTDIR/${method}.log"
    local -a args=(
        --task "$TASK"
        --lm_left "$left" --lm_right "$right"
        --batch_size "$BATCH_SIZE" --max_len "$MAX_LEN"
        --n_epochs "$EPOCHS" --lr "$LR" --bf16
    )
    if [[ "$mode" == zero_shot ]]; then
        args+=(--zero_shot --zero_shot_threshold "$ZERO_SHOT_THRESHOLD")
    else
        args+=(--loss contrastive --contrastive_margin "$MARGIN" --projection_dim "$PROJ_DIM")
    fi

    echo "[$(date)] $TASK | $method"
    python -u train_ditto.py "${args[@]}" 2>&1 | tee "$logfile"

    # Parse the standardized final lines emitted by train_ditto.py.
    # Refuse to silently write partial results.
    local val test
    val="$(awk '/^val f1 = / {v=$4} END {print v}' "$logfile")"
    test="$(awk '/^test f1 = / {v=$4} END {print v}' "$logfile")"
    if [[ -z "$val" || -z "$test" ]]; then
        echo "Missing final metrics: $logfile" >&2
        exit 1
    fi
    local margin="" threshold="" is_zero_shot="false"
    if [[ "$mode" == zero_shot ]]; then
        threshold="$ZERO_SHOT_THRESHOLD"
        is_zero_shot="true"
    else
        margin="$MARGIN"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$TASK" "$method" "$left" "$right" "$is_zero_shot" \
        "$margin" "$threshold" "$val" "$test" >> "$CSV"
}

#run_method zero_shot_roberta          roberta    roberta    zero_shot
#run_method zero_shot_distilbert       distilbert distilbert zero_shot
#run_method zero_shot_roberta_distilbert  roberta    distilbert zero_shot
#run_method contrastive_bert    distilbert distilbert train
#run_method contrastive_roberta       roberta    roberta    train
run_method contrastive_roberta_distilbert bert  roberta train
#run_method contrastive_roberta_distilbert bert  distilbert train
#run_method contrastive_roberta_distilbert roberta  distilbert train

echo "Completed $TASK: $CSV"
cat "$CSV"
