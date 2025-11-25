#!/bin/bash


device=cpu

# RLM parameters (fixed)
dataset=rlm
num_features=128       # Vocabulary size
num_layers=1          # Tree depth (generates 2^L leaves)
num_tokens=$((2**num_layers))  # Sequence length (automatically calculated)

# Beta values to test
beta_values=(0.6 0.8 1.2 1.8 2.4 3.0)

# Number of repetitions per beta
num_repetitions=8

# Training parameters (fixed)
train_size=$((2**15))
batch_size=64
accumulation=1
test_size=$((2**12))
input_format=long
whitening=0

# Model architecture 
model=transformer_mlm
depth=1
embedding_dim=128
num_heads=4
ffwd_size=128
dropout=0.0

# Optimizer settings 
optim=sgd
lr=1e-3
momentum=0.9

# Learning rate scheduler
scheduler=""
warmup_time=8
decay_time=$((2**18))

# Training control 
max_epochs=1
max_iters=5000
print_freq=1024
save_freq=2
measure_train=FALSE

# Output directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
results_dir="${SCRIPT_DIR}/results/beta_experiments"

# Create results directory if it doesn't exist
mkdir -p "${results_dir}"

# Add current directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "=========================================="
echo "Beta Experiment Suite"
echo "=========================================="
echo "Beta values: ${beta_values[@]}"
echo "Repetitions per beta: ${num_repetitions}"
echo "Total runs: $((${#beta_values[@]} * num_repetitions))"
echo "Results directory: ${results_dir}"
echo "=========================================="
echo ""

# Counter for total experiments
total_experiments=$((${#beta_values[@]} * num_repetitions))
current_experiment=0

# Loop over beta values
for beta in "${beta_values[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing beta = ${beta}"
    echo "=========================================="
    
    # Loop over repetitions
    for rep in $(seq 1 ${num_repetitions}); do
        current_experiment=$((current_experiment + 1))
        
        echo ""
        echo ">>> Experiment ${current_experiment}/${total_experiments}: beta=${beta}, repetition=${rep}"
        
        # Generate random seeds for this repetition
        seed_rules=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
        seed_sample=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
        seed_model=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
        
        # Output filename
        outname="${results_dir}/rlm_v${num_features}_L${num_layers}_beta${beta}_rep${rep}"
        
        echo "    Seeds: rules=${seed_rules}, sample=${seed_sample}, model=${seed_model}"
        echo "    Output: ${outname}"
        
        # Run training
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
        
        echo "    ✓ Completed experiment ${current_experiment}/${total_experiments}"
    done
    
    echo ""
    echo "✓ Completed all repetitions for beta = ${beta}"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in: ${results_dir}"
echo "Total experiments run: ${total_experiments}"
echo ""
echo "To analyze results, you can load the .pt files in Python:"
echo "  import torch"
echo "  data = torch.load('${results_dir}/rlm_v${num_features}_L${num_layers}_beta2.0_rep1.pt')"
echo ""
