#!/bin/bash

#SBATCH --job-name=embeddings
#SBATCH -o ./logs/embeddings.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00

cd /cluster/tufts/c26sp1cs0145/mzubrz01/hvm/ || exit 1

pixi run python create_embeddings.py