#!/bin/bash

#SBATCH --job-name upgrade_test
#SBATCH -o ./logs/upgrade_test.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p lulab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=1-00:00:00

pixi run python run_hvm.py \
  --feature_file sts17_octen_features.npz \
  --hidden_sizes 30 \
  --n_iters 1500 \
  --batch_size 32 \
  --n_mc_samples 3 \
  --flow_length 1 \
  --step_size 5e-5 \
  --use_sigmoid_output True \
  --n_valid_samples 25 \
  --n_test_samples 50 \
  --print_every 100
