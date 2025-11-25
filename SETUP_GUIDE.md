# Setup and Quick Start Guide

## Project Created Successfully! ✓

Your new `RLM_transformer_training` project is ready for next-token prediction training.

## What Was Created

```
RLM_transformer_training/
├── code.sh                          ✓ Training script (executable)
├── README.md                        ✓ User documentation
├── PROJECT_SUMMARY.md               ✓ Technical summary
├── SETUP_GUIDE.md                   ✓ This file
├── RLM_clean/
│   ├── __init__.py                  ✓
│   ├── main.py                      ✓ Training entry point
│   ├── init.py                      ✓ Initialization functions
│   ├── measures.py                  ✓ Evaluation metrics
│   ├── datasets/
│   │   ├── __init__.py              ✓
│   │   └── random_language_model.py ✓ RLM data generation
│   └── models/
│       ├── __init__.py              ✓
│       ├── transformer.py           ✓ Transformer models
│       └── fcn.py                   ✓ MLP layers
├── main_test.ipynb                  ✓ Testing notebook
└── rlm_data_generation.ipynb        ✓ Data exploration notebook
```

## Quick Start

### 1. Navigate to Project Directory
```bash
cd RLM_transformer_training
```

### 2. Run Training
```bash
bash code.sh
```

That's it! Training will start with the default configuration.

## What to Expect

### Training Output
```
Device: cpu
Training for 30000 steps (online regime)
Task: Next-token prediction
Sequence length: 2 tokens
Vocabulary size: 128
Optimizer: sgd (lr=0.001, momentum=0.9)
# parameters: XXXXX

Initial test loss: X.XXXX, test acc: X.XXXX
RLM entropy: X.XXXX
RLM marginal: X.XXXX

step:   1024 | running loss: X.XXXX | test loss: X.XXXX | test acc: X.XXXX
step:   2048 | running loss: X.XXXX | test loss: X.XXXX | test acc: X.XXXX
...
```

### Output Files
Results will be saved to:
```
results/rlm_128L1_b2_ts32768_1.pt
results/rlm_128L1_b2_ts32768_1_config.pt
results/rlm_128L1_b2_ts32768_1_t{step}.pt
```

## Configuration

### Default Settings (in code.sh)
- **Vocabulary**: 128 tokens
- **Tree Depth**: 1 (generates 2 leaves)
- **Sequence Length**: 2 tokens (input: 1, predict: 1)
- **Beta**: 2.0
- **Optimizer**: SGD with momentum=0.9
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Max Iterations**: 30,000

### Modify Configuration
Edit `code.sh` to change parameters:
```bash
# Example: Larger vocabulary and longer training
num_features=256
max_iters=50000
beta=3.0
```

## Next Steps

### 1. Monitor Training
Watch the console output for:
- Running loss (should decrease)
- Test loss (should approach RLM entropy)
- Test accuracy (should increase)

### 2. Analyze Results
Open `main_test.ipynb` in Jupyter:
```bash
jupyter notebook main_test.ipynb
```

### 3. Explore Data
Open `rlm_data_generation.ipynb`:
```bash
jupyter notebook rlm_data_generation.ipynb
```

## Understanding the Task

### Next-Token Prediction
Given a sequence, predict the next token:

**Example with L=1 (2 leaves):**
- Input: `[L1]` → Output: `L2`

**Example with L=3 (8 leaves):**
- Input: `[L1, L2, L3, L4, L5, L6, L7]` → Output: `L8`

### How It Works
1. RLM generates sequences from hierarchical grammar
2. Model sees all tokens except the last
3. Model predicts only the last token
4. Uses causal attention (can't look ahead)

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project directory:
```bash
cd RLM_transformer_training
python RLM_clean/main.py --help
```

### Device Issues
- Default device: `cpu`
- For GPU: Edit `code.sh` and change `device=cpu` to `device=cuda`
- For Apple Silicon: Change to `device=mps`

### Memory Issues
If you run out of memory:
- Reduce `batch_size` in `code.sh`
- Reduce `embedding_dim`
- Reduce `num_features` (vocabulary size)

## Key Differences from Original RLM

This implementation is specifically designed for **next-token prediction**:

| Feature | Original RLM | This Project |
|---------|-------------|--------------|
| Task | Root from leaves | Next token |
| Attention | Bidirectional | Causal |
| Output | Classification | Single token |
| Optimizer | Adam | SGD + momentum |

## Documentation

- **README.md**: User-friendly documentation
- **PROJECT_SUMMARY.md**: Technical details and design decisions
- **SETUP_GUIDE.md**: This file - quick start guide

## Testing the Installation

Run a quick test:
```bash
cd RLM_transformer_training
python -c "import RLM_clean.datasets as datasets; print('✓ Import successful')"
```

If you see `✓ Import successful`, you're ready to go!

## Getting Help

1. Check **README.md** for usage examples
2. Check **PROJECT_SUMMARY.md** for technical details
3. Look at the notebooks for interactive examples
4. Review the code comments in source files

## Ready to Train!

Everything is set up. Just run:
```bash
bash code.sh
```

Happy training! 🚀
