"""
Main training script for next-token prediction with RLM.

This script trains a transformer model to predict the next token in a sequence
generated from a Random Language Model (RLM) grammar.
"""

import os
import sys
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import math
import random
import argparse

import RLM_files.datasets as datasets
import RLM_files.models as models
import RLM_files.init as init
import RLM_files.measures as measures


def run(config):
    """
    Main training loop for next-token prediction.
    
    Args:
        config: Configuration object with all hyperparameters
        
    Returns:
        dynamics: Training dynamics (loss, accuracy over time)
    """
    config.input_size = 2**config.num_layers
    if not hasattr(config, 'max_epochs'):
        config.max_epochs = 1

    print(f"Device: {config.device}")
    print(f"Training for {config.max_iters} steps (online regime)")
    print(f"Task: Next-token prediction")
    print(f"Sequence length: {config.num_tokens} tokens")
    print(f"Vocabulary size: {config.num_features}")
    print(f"Optimizer: {config.optim} (lr={config.lr}, momentum={config.momentum})")

    # ===== Create ONE RLM for both training and testing (true online learning) =====
    rlm = datasets.RLM(
        v=config.num_features,
        L=config.num_layers,
        beta=config.beta,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample,  # Same RLM for train and test
        num_data=None,
        probs=None,
        transform=None
    )
    
    # No fixed test set - will generate fresh data at each checkpoint
    test_loader = None

    # ===== Initialize model =====
    model = init.init_model(config)
    model0 = copy.deepcopy(model)

    criterion, optimizer, scheduler = init.init_training(model, config)
    print_ckpts, save_ckpts = init.init_loglinckpt(
        config.print_freq, config.max_iters, freq=config.save_freq
    )
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    step = 0
    # Generate initial fresh test data from the SAME RLM
    test_inputs_init, test_targets_init = rlm.sample_batch(batch_size=config.batch_size**2, L=config.num_layers)
    test_inputs_init = init.transform_inputs(test_inputs_init, config).to(config.device)
    test_targets_init = test_targets_init.to(config.device)
    
    model.eval()
    with torch.no_grad():
        test_outputs_init = model(test_inputs_init)
        if len(test_outputs_init.shape) == 3:
            test_outputs_init = test_outputs_init[:, -1, :]
        testloss_init = criterion(test_outputs_init, test_targets_init).item()
        test_predictions_init = torch.argmax(test_outputs_init, dim=-1)
        testacc_init = (test_predictions_init == test_targets_init).float().mean().item()
    model.train()
    
    dynamics = [{'t': 0, 'testloss': testloss_init, 'testacc': testacc_init}]
    best = {'step': 0, 'model': None, 'loss': testloss_init}

    # Save initial configuration
    if config.checkpoints:
        torch.save(
            {'config': config, 'rules': rlm.M},
            f"{config.outname}_config.pt"
        )
        output = {
            'model': copy.deepcopy(model.state_dict()),
            'state': dynamics[-1],
            'step': step
        }
        torch.save(output, f"{config.outname}_t{step}.pt")

    running_loss = 0.0
    test_loss = dynamics[-1]['testloss']
    test_acc = dynamics[-1]['testacc']

    print(f"\nInitial test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
    print(f"RLM entropy: {rlm.entropy:.4f}")
    print(f"RLM marginal: {rlm.marginal:.4f}\n")

    # ===== Training loop =====
    while step < config.max_iters:
        # Generate online batch for next-token prediction from the SAME RLM
        inputs_raw, targets_raw = rlm.sample_batch(
            batch_size=config.batch_size,
            L=config.num_layers
        )

        inputs_raw = inputs_raw.to(config.device)
        targets_raw = targets_raw.to(config.device)

        # Transform inputs
        inputs = init.transform_inputs(inputs_raw, config)

        # Forward pass
        outputs = model(inputs)  # shape: (batch_size, seq_len, vocab_size)
        # For next-token prediction, use only the last position
        outputs = outputs[:, -1, :]  # shape: (batch_size, vocab_size)
        loss = criterion(outputs, targets_raw)  # targets_raw shape: (batch_size,)
        running_loss += loss.item()

        # Backward pass with gradient accumulation
        loss = loss / config.accumulation
        loss.backward()

        if (step + 1) % config.accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        # Logging and checkpointing
        if step == print_ckpt:
            # Generate FRESH test data from the SAME RLM (truly online evaluation)
            test_inputs, test_targets = rlm.sample_batch(batch_size=config.test_size, L=config.num_layers)
            test_inputs = init.transform_inputs(test_inputs, config).to(config.device)
            test_targets = test_targets.to(config.device)
            
            # Evaluate on fresh test data
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                if len(test_outputs.shape) == 3:
                    test_outputs = test_outputs[:, -1, :]
                test_loss = criterion(test_outputs, test_targets).item()
                test_predictions = torch.argmax(test_outputs, dim=-1)
                test_acc = (test_predictions == test_targets).float().mean().item()
            model.train()
            
            if test_loss < best['loss']:
                best['step'] = step
                best['loss'] = test_loss
                best['model'] = copy.deepcopy(model.state_dict())

            print(
                f'step: {step:6d} | '
                f'running loss: {running_loss / step:06.4f} | '
                f'test loss: {test_loss:06.4f} | '
                f'test acc: {test_acc:06.4f}'
            )
            print_ckpt = next(print_ckpts)

            # Save checkpoint
            if step >= save_ckpt:
                print(f'Checkpoint at step {step}, saving data ...')
                save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
                dynamics.append(save_dict)

                if config.checkpoints:
                    output = {
                        'model': copy.deepcopy(model.state_dict()),
                        'state': dynamics[-1],
                        'step': step
                    }
                    torch.save(output, f"{config.outname}_t{step}.pt")
                else:
                    output = {
                        'entropy': rlm.entropy,
                        'marginal': rlm.marginal,
                        'dynamics': dynamics,
                        'step': step
                    }
                    torch.save({'config': config, 'output': output}, f"{config.outname}.pt")
                
                save_ckpt = next(save_ckpts)

        # Early stopping based on loss threshold
        if (running_loss / step) <= config.loss_threshold:
            print(f"\nReached loss threshold at step {step}")
            save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
            dynamics.append(save_dict)

            if config.checkpoints:
                output = {
                    'model': copy.deepcopy(model.state_dict()),
                    'state': dynamics[-1],
                    'step': step
                }
                torch.save(output, f"{config.outname}_t{step}.pt")
            else:
                output = {
                    'entropy': rlm.entropy,
                    'marginal': rlm.marginal,
                    'dynamics': dynamics,
                    'step': step
                }
                torch.save({'config': config, 'output': output}, f"{config.outname}.pt")
            break

    print(f"\nTraining complete!")
    print(f"Final RLM entropy: {rlm.entropy:.4f}")
    print(f"Final RLM marginal: {rlm.marginal:.4f}")
    print(f"Best test loss: {best['loss']:.4f} at step {best['step']}")
    
    return dynamics


def default_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    
    parser = argparse.ArgumentParser(
        description='Next-token prediction with Random Language Model'
    )
    
    # Device and dataset
    parser.add_argument('--device', type=str, default=default_device(),
                       help='Device to use (cuda/mps/cpu)')
    parser.add_argument('--dataset', type=str, default='rlm',
                       help='Dataset type')
    
    # RLM parameters
    parser.add_argument('--num_features', type=int, default=128,
                       help='Vocabulary size')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Tree depth (generates 2^L leaves)')
    parser.add_argument('--seed_rules', type=int, default=12345678,
                       help='Seed for grammar generation')
    parser.add_argument('--num_tokens', type=int, default=2,
                       help='Sequence length (2^num_layers)')
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Temperature parameter for RLM')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--test_size', type=int, default=2**12,
                       help='Test set size')
    parser.add_argument('--seed_sample', type=int, default=56781234,
                       help='Seed for data sampling')
    parser.add_argument('--input_format', type=str, default='long',
                       help='Input format (long/onehot)')
    parser.add_argument('--whitening', type=int, default=0,
                       help='Apply whitening to inputs')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='transformer_mlm',
                       help='Model type')
    parser.add_argument('--depth', type=int, default=1,
                       help='Number of transformer layers')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=1,
                       help='Number of attention heads')
    parser.add_argument('--ffwd_size', type=int, default=4,
                       help='Feedforward size multiplier')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')
    parser.add_argument('--seed_model', type=int, default=12345678,
                       help='Seed for model initialization')
    
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default='sgd',
                       help='Optimizer (sgd/adam)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--scheduler', type=str, default='cosine-warmup',
                       help='LR scheduler')
    parser.add_argument('--warmup_time', type=int, default=8,
                       help='Warmup steps')
    parser.add_argument('--decay_time', type=int, default=2**18,
                       help='Decay steps')
    
    # Training control
    parser.add_argument('--max_epochs', type=int, default=1,
                       help='Maximum epochs (not used in online regime)')
    parser.add_argument('--max_iters', type=int, default=30000,
                       help='Maximum training iterations')
    parser.add_argument('--print_freq', type=int, default=1024,
                       help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                       help='Save frequency (log scale)')
    parser.add_argument('--measure_train', action='store_true', default=False,
                       help='Measure training metrics')
    parser.add_argument('--loss_threshold', type=float, default=1e-3,
                       help='Early stopping loss threshold')
    parser.add_argument('--checkpoints', default=False, action='store_true',
                       help='Save model checkpoints')
    
    # Output
    parser.add_argument('--outname', type=str, required=True,
                       help='Output file path')
    
    config = parser.parse_args()
    
    # Run training
    dynamics = run(config)
