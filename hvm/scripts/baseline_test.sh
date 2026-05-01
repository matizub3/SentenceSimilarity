#!/bin/bash

#SBATCH --job-name baseline_test
#SBATCH -o ./logs/baseline_test.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=1-00:00:00

pixi run python run_baseline.py \
  --feature_file sts17_octen_features.npz \
  --hidden_sizes 100 \
  --n_iters 2000 \
  --batch_size 32 \
  --n_mc_samples 50 \
  --print_every 100
