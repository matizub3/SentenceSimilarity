#!/bin/bash -l
#SBATCH --job-name baseline_hparam_search
#SBATCH -o ./logs/baseline_hparam_search_%A_%a.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=0-00:30:00
#SBATCH --array=0-431%8       # must match scripts/run_baseline_hparam_job.sh GRID size (432)

set -euo pipefail

# Slurm batch shells often do not inherit interactive PATH.
if ! command -v pixi >/dev/null 2>&1; then
  if [ -x "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
  fi
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found in PATH in batch environment"
  echo "PATH=$PATH"
  exit 127
fi

pixi run python scripts/run_baseline_hparam_job.sh --task_id $SLURM_ARRAY_TASK_ID