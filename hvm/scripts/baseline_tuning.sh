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
  --hidden_sizes 100 \
  --n_iters 2000 \
  --batch_size 64 \
  --n_mc_samples 5 \
  --step_size 1e-3 \
  --likelihood_stddev 0.10 \
  --init_q_stddev 0.05 \
  --print_every 100
