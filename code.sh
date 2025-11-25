#!/bin/bash

# Configuration for last-token prediction training with RLM

# Device
device=cpu

# RLM parameters
dataset=rlm
num_features=64       # Vocabulary size
num_layers=1            # Tree depth (generates 2^L leaves)
num_tokens=$((2**num_layers))  # Sequence length (automatically calculated as 2^num_layers)
beta=2                  # Temperature parameter

# Training parameters
train_size=$((2**15))   # Not used in online regime, kept for naming
batch_size=64
accumulation=1
test_size=$((2**12))  # 4096 samples for test (much faster)
input_format=long
whitening=0

# Model architecture
model=transformer_mlm
depth=1                 # Number of transformer layers
embedding_dim=128
num_heads=4
ffwd_size=128
dropout=0.0

# Optimizer settings
optim=sgd               # Using SGD
lr=1e-3
momentum=0.9            # SGD momentum

# Learning rate scheduler (empty string = no scheduler, constant LR)
scheduler=""
warmup_time=8
decay_time=$((2**18))

# Training control
max_epochs=1            # Not relevant in online regime
max_iters=5000          # Main stopping criterion (matching notebook)
print_freq=1024
save_freq=2
measure_train=FALSE

# Seeds
seed_rules=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
seed_sample=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
seed_model=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')

# Output - results folder inside RLM_transformer_training
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
results_dir="${SCRIPT_DIR}/results"

# Create results directory if it doesn't exist
mkdir -p "${results_dir}"

outname="${results_dir}/rlm_${num_features}L${num_layers}_b${beta}_ts${train_size}_1"

# Run training
# Add current directory to PYTHONPATH so RLM_files can be imported
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python RLM_files/main.py \
  --device "$device" \
  --dataset "$dataset" \
  --num_features "$num_features" \
  --num_layers "$num_layers" \
  --seed_rules "$seed_rules" \
  --num_tokens "$num_tokens" \
  --beta "$beta" \
  --batch_size "$batch_size" \
  --accumulation "$accumulation" \
  --test_size "$test_size" \
  --seed_sample "$seed_sample" \
  --input_format "$input_format" \
  --whitening "$whitening" \
  --model "$model" \
  --depth "$depth" \
  --embedding_dim "$embedding_dim" \
  --num_heads "$num_heads" \
  --ffwd_size "$ffwd_size" \
  --dropout "$dropout" \
  --seed_model "$seed_model" \
  --optim "$optim" \
  --lr "$lr" \
  --momentum "$momentum" \
  --scheduler "$scheduler" \
  --warmup_time "$warmup_time" \
  --decay_time "$decay_time" \
  --max_epochs "$max_epochs" \
  --max_iters "$max_iters" \
  --print_freq "$print_freq" \
  --save_freq "$save_freq" \
  --outname "$outname"
