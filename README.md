# RLM Transformer Training - Next-Token Prediction

This project trains transformer models on next-token prediction using sequences generated from a Random Language Model (RLM).

## Overview

The Random Language Model (RLM) generates sequences of tokens through a hierarchical grammar. This project trains a transformer to predict the next token in a sequence given previous tokens.

### Task: Next-Token Prediction

Given a sequence of tokens `[L1, L2, L3, ..., Ln]`, the model predicts the next token `L(n+1)`.

**Example:**
- Input: `[L1, L2, L3, L4, L5, L6, L7]` (7 tokens)
- Output: `L8` (single token prediction)

The model uses causal attention to process the sequence and only predicts the final next token.

## Project Structure

```
RLM_transformer_training/
├── code.sh                          # Main training script
├── RLM_clean/
│   ├── main.py                      # Training entry point
│   ├── init.py                      # Initialization functions
│   ├── measures.py                  # Evaluation metrics
│   ├── datasets/
│   │   └── random_language_model.py # RLM data generation
│   └── models/
│       ├── transformer.py           # Transformer models
│       └── fcn.py                   # MLP layers
├── main_test.ipynb                  # Testing notebook
├── rlm_data_generation.ipynb        # RLM data exploration
└── README.md                        # This file
```

## Key Features

### Model Architecture
- **Causal Transformer**: Uses decoder-only architecture with causal attention
- **Single Token Output**: Predicts only the next token (not full sequence)
- **Configurable**: Adjustable depth, embedding dimension, attention heads

### Training Setup
- **Optimizer**: SGD with momentum (default: momentum=0.9)
- **Learning Rate**: 1e-3 with cosine warmup scheduler
- **Online Regime**: Generates fresh batches from RLM grammar at each step
- **Batch Size**: 64 (configurable)

### RLM Parameters
- **Vocabulary Size** (`num_features`): Number of unique tokens
- **Tree Depth** (`num_layers`): Determines sequence length (2^L leaves)
- **Beta** (`beta`): Temperature parameter controlling grammar randomness

## Usage

### Basic Training

Run the default configuration:

```bash
bash code.sh
```

### Custom Configuration

Edit `code.sh` to modify parameters:

```bash
# Model size
embedding_dim=64
num_heads=1
depth=1

# Training
batch_size=64
max_iters=30000
lr=1e-3
momentum=0.9

# RLM
num_features=128  # Vocabulary size
beta=2.0          # Temperature
```

### Python API

```python
import torch
from RLM_clean import datasets, models, init

# Create RLM
rlm = datasets.RLM(
    v=128,           # vocab size
    L=1,             # tree depth
    beta=2.0,        # temperature
    seed_rules=12345,
    seed_samples=67890,
    num_data=None,
    probs=None,
    transform=None
)

# Generate batch for next-token prediction
inputs, targets = rlm.sample_batch(batch_size=64, L=1)
# inputs shape: (64, 1)  - sequence of 1 token
# targets shape: (64,)   - next token to predict
```

## Configuration Parameters

### RLM Parameters
- `num_features`: Vocabulary size (default: 128)
- `num_layers`: Tree depth, generates 2^L leaves (default: 1)
- `beta`: Temperature parameter (default: 2.0)
- `num_tokens`: Sequence length, should equal 2^num_layers (default: 2)

### Model Parameters
- `model`: Model type (default: `transformer_mlm`)
- `depth`: Number of transformer layers (default: 1)
- `embedding_dim`: Embedding dimension (default: 64)
- `num_heads`: Number of attention heads (default: 1)
- `ffwd_size`: Feedforward multiplier (default: 4)
- `dropout`: Dropout probability (default: 0.0)

### Training Parameters
- `optim`: Optimizer type - `sgd` or `adam` (default: `sgd`)
- `lr`: Learning rate (default: 1e-3)
- `momentum`: SGD momentum (default: 0.9)
- `scheduler`: LR scheduler (default: `cosine-warmup`)
- `warmup_time`: Warmup steps (default: 8)
- `decay_time`: Decay steps (default: 262144)
- `batch_size`: Batch size (default: 64)
- `max_iters`: Maximum training iterations (default: 30000)

## Output

Training produces:
- **Checkpoints**: Model states at logarithmically spaced intervals
- **Dynamics**: Training/test loss and accuracy over time
- **Final Model**: Best model based on test loss

Output files are saved to the path specified in `outname`:
```
results/rlm_128L1_b2_ts32768_1.pt
results/rlm_128L1_b2_ts32768_1_config.pt
results/rlm_128L1_b2_ts32768_1_t{step}.pt
```

## Notebooks

### main_test.ipynb
Testing and analysis notebook for:
- Loading trained models
- Evaluating performance
- Visualizing learning curves
- Testing predictions

### rlm_data_generation.ipynb
RLM data exploration notebook for:
- Understanding RLM grammar
- Visualizing tree structures
- Analyzing token distributions
- Exploring sequence properties

## Differences from Original RLM

This implementation differs from the original RLM project:

1. **Task**: Next-token prediction instead of root prediction from leaves
2. **Attention**: Causal (decoder) attention instead of bidirectional
3. **Output**: Single token prediction instead of classification
4. **Optimizer**: SGD with momentum instead of Adam
5. **Data**: Sequences of leaves for autoregressive prediction

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Jupyter (for notebooks)

## Citation

If you use this code, please cite the original RLM work and acknowledge this implementation.

## License

[Specify license here]
