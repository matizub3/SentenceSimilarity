#!/bin/bash
#SBATCH --job-name baseline_hparam_search
#SBATCH -o ./logs/baseline_hparam_search_%A_%a.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=0-00:30:00
#SBATCH --array=0-31%8        # 32 jobs, max 8 running at once

pixi run python scripts/run_baseline_hparam_job.sh --task_id $SLURM_ARRAY_TASK_ID