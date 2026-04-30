#!/bin/bash

#SBATCH --job-name baseline_test
#SBATCH -o ./logs/baseline_test.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p lulab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=1-00:00:00

pixi run python run_baseline.py \
  --feature_file sts17_octen_features.npz \
  --hidden_sizes 30 \
  --n_iters 10 \
  --batch_size 8 \
  --n_mc_samples 1 \
  --print_every 1
