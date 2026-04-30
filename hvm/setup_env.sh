#!/bin/bash

export PROJECT_DIR=/cluster/tufts/c26sp1cs0145/mzubrz01/hvm

mkdir -p $PROJECT_DIR/.pixi-cache
mkdir -p $PROJECT_DIR/.hf-cache/datasets

export PIXI_CACHE_DIR=$PROJECT_DIR/.pixi-cache
export RATTLER_CACHE_DIR=$PROJECT_DIR/.pixi-cache
export HF_HOME=$PROJECT_DIR/.hf-cache
export TRANSFORMERS_CACHE=$PROJECT_DIR/.hf-cache
export HF_DATASETS_CACHE=$PROJECT_DIR/.hf-cache/datasets
