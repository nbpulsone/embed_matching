#!/usr/bin/env bash
# Submit one SLURM job per entity-matching dataset.
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
mkdir -p logs/embed_matching_hetero_noproj

for task in 'Structured/Beer' 'Textual/Abt-Buy' 'Dirty/iTunes-Amazon'; do
    tag="${task//\//_}"
    echo "Submitting $task"
    sbatch \
        --job-name="em_${tag}" \
        --output="logs/embed_matching_hetero_noproj/${tag}_%j.out" \
        --error="logs/embed_matching_hetero_noproj/${tag}_%j.err" \
        run_embed_matching_dataset.sh "$task"
done
