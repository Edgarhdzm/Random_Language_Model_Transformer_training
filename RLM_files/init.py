import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import RLM_files.datasets as datasets
import RLM_files.models as models
import RLM_files.measures as measures


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_time: Number of warmup steps
        decay_time: Total number of steps for decay
        min_lr_factor: Minimum learning rate as fraction of initial LR
    """
    def __init__(self, optimizer, warmup_time, decay_time, min_lr_factor):
        self.warmup = warmup_time
        self.decay = decay_time
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.warmup:
            # Linear warmup from 0 to 1
            return step / self.warmup
        elif step < self.decay:
            # Cosine decay to min_lr_factor
            decay_step = step - self.warmup
            total_decay = self.decay - self.warmup
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / total_decay))
            return self.min_lr_factor + (1 - self.min_lr_factor) * cosine_decay
        else:
            # Stay constant after decay
            return self.min_lr_factor


def transform_inputs(inputs, args):
    """
    Transform input tokens for the model.
    
    For next-token prediction with transformers, we use 'long' format
    (token indices) which are directly embedded by the model.
    
    Args:
        inputs: Token indices (batch_size, seq_len)
        args: Configuration arguments
        
    Returns:
        Transformed inputs ready for the model
    """
    B = inputs.shape[0]

    if args.num_tokens < args.input_size:
        # Only take last num_tokens positions
        inputs = inputs[:, -args.num_tokens:]

    if 'onehot' not in args.input_format:
        assert not args.whitening, "Whitening only implemented for one-hot encoding"

    if 'onehot' in args.input_format:
        inputs = F.one_hot(
            inputs.long(),
            num_classes=args.num_features
        ).float()  # size B, T, C

        if args.whitening:
            inv_sqrt_norm = (1.-1./args.num_features) ** -.5
            inputs = (inputs - 1./args.num_features) * inv_sqrt_norm

        inputs = inputs.permute(0, 2, 1)  # size B, C, T

        if 'fcn' in args.model:
            # FCN requires flattening of the input
            inputs = inputs.transpose(1,2).flatten(start_dim=1)

        if 'transformer' in args.model:
            # Transformer requires B,T,C input format
            inputs = inputs.transpose(1,2)

    elif 'long' in args.input_format:
        # For next-token prediction, just convert to long
        # No masking needed - causal attention handles it
        inputs = inputs.long()

    else:
        raise ValueError(f'format argument {args.input_format} is invalid!')

    return inputs


def init_model(args):
    """
    Initialize the transformer model for next-token prediction.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Initialized model
    """
    torch.manual_seed(args.seed_model)

    if 'transformer' in args.model:
        assert args.num_heads is not None, 'transformer model requires argument num_heads!'
        assert args.embedding_dim is not None, 'transformer model requires argument embedding_dim!'

        if args.ffwd_size is None:
            args.ffwd_size = 4

        if args.model == 'transformer_mlm':
            # Next-token prediction model with causal attention
            model = models.MLM(
                vocab_size=args.num_features,
                block_size=args.num_tokens,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                ffwd_size=args.ffwd_size,
                num_layers=args.depth,
                dropout=args.dropout
            )
        elif args.model == 'transformer_clm':
            model = models.CLM(
                vocab_size=args.num_features,
                block_size=args.num_tokens,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                ffwd_size=args.ffwd_size,
                num_layers=args.depth,
                dropout=args.dropout,
                share_emb=False,
            )
        else:
            raise ValueError(f'transformer model {args.model} not supported!')
    else:
        raise ValueError('Only transformer models supported in this version!')

    model = model.to(args.device)
    param_count = sum([p.numel() for p in model.parameters()])
    print("# parameters:", param_count)

    return model


def init_training(model, args):
    """
    Initialize training components (optimizer, scheduler, loss).
    
    Args:
        model: The model to train
        args: Configuration arguments
        
    Returns:
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
    """
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.optim == 'adam':
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.
        )
    else:
        raise ValueError("optimizer is invalid (sgd, adam)!")

    if args.scheduler is None or args.scheduler == '':
        # No scheduler - constant learning rate (matching notebook)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.max_iters
        )
    elif args.scheduler == 'cosine':
        assert args.decay_time is not None, 'cosine scheduler requires argument decay_time!'
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.decay_time, eta_min=0.1*args.lr
        )
    elif args.scheduler == 'cosine-warmup':
        assert args.warmup_time is not None, 'cosine-warmup scheduler requires argument warmup_time!'
        assert args.decay_time is not None, 'cosine-warmup scheduler requires argument decay_time!'
        scheduler = CosineWarmupLR(
            optimizer, args.warmup_time, args.decay_time, 0.1
        )
    else:
        raise ValueError(f"scheduler {args.scheduler} is invalid!")

    return criterion, optimizer, scheduler


def init_output(model, criterion, train_loader, test_loader, args):
    """
    Initialize output tracking for the experiment.
    
    Args:
        model: The model
        criterion: Loss function
        train_loader: Training data loader (can be None for online regime)
        test_loader: Test data loader
        args: Configuration arguments
        
    Returns:
        dynamics: List to track training dynamics
        best: Dictionary to track best model
    """
    if test_loader is not None:
        testloss, testacc = measures.test(model, criterion, test_loader, args.device)
    else:
        testloss, testacc = float('nan'), float('nan')

    print_dict = {'t': 0, 'testloss': testloss, 'testacc': testacc}

    if args.measure_train and train_loader is not None:
        trainloss, trainacc = measures.test(model, criterion, train_loader, args.device)
        print_dict['trainloss'] = trainloss
        print_dict['trainacc'] = trainacc

    dynamics = [print_dict]
    best = {'step': 0, 'model': None, 'loss': testloss}

    return dynamics, best


def log2ckpt(end, freq):
    """
    Initialize log-spaced iterator.

    Returns:
        List with integer steps spaced multiplicatively by 2**(1/freq) until end.
    """
    current = 1.
    factor = 2**(1./freq)
    threshold = 2**(math.ceil(math.log(1./(factor-1)))+1)
    checkpoints = []

    while current < threshold:
        checkpoints.append(round(current))
        current += 1

    while round(current) < end:
        checkpoints.append(round(current))
        current *= factor

    checkpoints.append(round(end))

    return checkpoints


def init_loglinckpt(step, end, freq):
    """
    Initialize checkpoint iterator.

    Returns:
        Two iterators, one for linear and one for logscale. The iterators coincide up to some multiple of step, 
        then one proceeds linearly in multiples of step and the other logarithmically in factors of 2**(1/freq).
    """
    # find the correct multiplier
    factor = 2**(1./freq)
    multiplier = 2**(math.ceil(math.log(1./(factor-1)))+1)

    # build log2ckpt lists until multiplier*step
    lin_ckpts = log2ckpt(multiplier*step, freq)
    log_ckpts = lin_ckpts.copy()

    # fill the linear list by adding steps until end
    current = lin_ckpts[-1] + step
    while current <= end:
        lin_ckpts.append(current)
        current += step
    lin_ckpts.append(0)

    # fill the log list by multiplying factors until end
    current = multiplier*factor
    while round(current)*step < end:
        log_ckpts.append(round(current)*step)
        current *= factor

    log_ckpts.append(round(end))
    log_ckpts.append(0)

    return iter(lin_ckpts), iter(log_ckpts)
