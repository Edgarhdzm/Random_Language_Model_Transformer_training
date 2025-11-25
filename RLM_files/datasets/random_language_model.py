import math
import random

import torch

import RLM_files.measures as measures


def rlm_make_M(V, beta, seed, device=None, generator=None, dtype=torch.float32):
    """
    Create random grammar rules for RLM.
    
    Args:
        V: Vocabulary size
        beta: Temperature parameter controlling rule randomness
        seed: Random seed for reproducibility
        device: Device to create tensors on
        generator: Random generator
        dtype: Data type for tensors
        
    Returns:
        M: Grammar rules tensor of shape (V, V, V)
           M[i,j,k] ~ P(children=(j,k) | parent=i)
    """
    random.seed(seed)
    device = device or torch.device('cpu')
    S = torch.randn((V, V, V), generator=generator, device=device, dtype=dtype) * math.sqrt(math.log(V))
    S = S * beta
    return torch.softmax(S.view(V, -1), dim=1).view(V, V, V)


def rlm_sample_trees(num_data, L, M, seed=None, prior=None, device=None, generator=None):
    """
    Sample trees from the RLM grammar.
    
    Args:
        num_data: Number of trees to sample
        L: Tree depth (generates 2^L leaves)
        M: Grammar rules
        seed: Random seed
        prior: Prior distribution over root tokens
        device: Device for tensors
        generator: Random generator
        
    Returns:
        trees: Dictionary mapping level -> tokens at that level
               trees[0] = root tokens
               trees[L] = leaf tokens
    """
    if seed is not None:
        random.seed(seed)

    device = device or M.device
    V = M.shape[0]
    trees = {}

    # Sample root tokens
    if prior is None:
        labels = torch.randint(V, (num_data,), device=device, generator=generator)
    else:
        p0 = prior.to(device=device, dtype=M.dtype)
        p0 = p0 / p0.sum()
        labels = torch.multinomial(p0, num_data, replacement=True, generator=generator)

    trees[0] = labels

    # Generate tree levels
    for l in range(1, L + 1):
        parents = trees[l - 1].reshape(-1)
        probs = M[parents].reshape(parents.numel(), -1)
        idx = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        x = idx // V
        y = idx % V
        trees[l] = torch.stack([x, y], dim=1).reshape(num_data, -1)

    return trees


class RLM:
    """
    Random Language Model for next-token prediction.
    
    Generates sequences of tokens from a hierarchical grammar and
    creates training data for next-token prediction tasks.
    """

    def __init__(
        self,
        v,              # vocabulary size
        L,              # number of layers (tree depth)
        beta,           # temperature parameter
        seed_rules,     # seed for grammar generation
        seed_samples,   # seed for data sampling
        num_data,       # not used in online regime
        probs,          # prior distribution (optional)
        transform=None,
        device=None,
        dtype=torch.float32,
    ):
        self.vocab_size = v
        self.L = L
        self.beta = beta
        self.seed_rules = seed_rules
        self.seed_samples = seed_samples
        self.transform = transform

        device = device or torch.device('cpu')

        # Generate grammar rules
        self.M = rlm_make_M(self.vocab_size, self.beta, self.seed_rules,
                            device=device, dtype=dtype)

        # Generator for online sampling
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed_samples)

        # No trees stored in memory for online regime
        self.trees = None

        # Estimate entropy and marginal with moderate sample size (analytical)
        num_for_meas = 2**15
        self.entropy = measures.conditional_entropy(self.M, self.vocab_size, num_for_meas)
        self.marginal = measures.marginal(self.M, self.vocab_size, num_for_meas)
        
        # Empirical estimates (computed on demand)
        self.empirical_entropy = None
        self.empirical_marginal = None

    def __len__(self):
        return 0

    def sample_batch(self, batch_size, L=None):
        """
        Sample a batch for next-token prediction training.
        
        Args:
            batch_size: Number of sequences to generate
            L: Tree depth (uses self.L if None)
            
        Returns:
            inputs: Sequence of tokens (batch_size, seq_len-1)
            targets: Next token to predict (batch_size,)
            
        Example:
            If leaves are [L1, L2, L3, L4, L5, L6, L7, L8]:
            inputs = [L1, L2, L3, L4, L5, L6, L7]
            targets = L8
        """
        if L is None:
            L = self.L

        trees = rlm_sample_trees(
            batch_size,
            L,
            self.M,
            seed=None,
            prior=None,
            device=self.M.device,
            generator=self.generator,
        )

        # Get full sequence of leaves
        full_sequence = trees[L]  # shape: (batch_size, 2^L)
        
        # For next-token prediction:
        # inputs: all tokens except last
        # targets: only the last token
        inputs = full_sequence[:, :-1]
        targets = full_sequence[:, -1]

        return inputs, targets

    def sample_eval_set(self, num_data, L=None):
        """
        Generate fixed evaluation set for next-token prediction.
        
        Args:
            num_data: Number of sequences to generate
            L: Tree depth (uses self.L if None)
            
        Returns:
            inputs: Sequence of tokens (num_data, seq_len-1)
            targets: Next token to predict (num_data,)
        """
        if L is None:
            L = self.L

        trees = rlm_sample_trees(
            num_data,
            L,
            self.M,
            seed=self.seed_samples,
            prior=None,
            device=self.M.device,
            generator=self.generator,
        )

        # Get full sequence of leaves
        full_sequence = trees[L]  # shape: (num_data, 2^L)
        
        # For next-token prediction:
        # inputs: all tokens except last
        # targets: only the last token
        inputs = full_sequence[:, :-1]
        targets = full_sequence[:, -1]

        return inputs, targets

    def compute_empirical_bigram_distribution(self, num_samples=100000):
        """
        Empirically compute the bigram distribution P(leaf_i, leaf_{i+1}) from generated sequences.
        
        This samples many trees and counts bigram occurrences to estimate the distribution.
        
        Args:
            num_samples: Number of sequences to generate for estimation
            
        Returns:
            Bigram probability matrix of shape (V, V) where entry [i,j] = P(leaf_t=i, leaf_{t+1}=j)
        """
        V = self.vocab_size
        bigram_counts = torch.zeros(V, V, device=self.M.device)
        total_bigrams = 0
        
        print(f"Computing empirical bigram distribution from {num_samples} samples...")
        
        for _ in range(num_samples):
            # Generate a tree
            trees = rlm_sample_trees(
                num_data=1,
                L=self.L,
                M=self.M,
                seed=None,
                prior=None,
                device=self.M.device,
                generator=self.generator,
            )
            
            # Get leaf sequence
            leaves = trees[self.L][0]  # Shape: (2^L,)
            
            # Count bigrams in this sequence
            for i in range(len(leaves) - 1):
                bigram_counts[leaves[i], leaves[i+1]] += 1
                total_bigrams += 1
        
        # Normalize to get probabilities
        bigram_probs = bigram_counts / total_bigrams if total_bigrams > 0 else bigram_counts
        
        return bigram_probs
    
    def compute_empirical_entropy(self, bigram_probs=None, num_samples=100000):
        """
        Compute the conditional entropy H(leaf_{t+1} | leaf_t) from empirical bigram distribution.
        
        This is the theoretical minimum loss for next-token prediction!
        
        Formula: H(X_{t+1} | X_t) = -∑_{x_t} P(x_t) ∑_{x_{t+1}} P(x_{t+1}|x_t) log P(x_{t+1}|x_t)
        
        Args:
            bigram_probs: Pre-computed bigram probabilities (if None, will compute)
            num_samples: Number of samples for empirical estimation
            
        Returns:
            Conditional entropy H(leaf_{t+1} | leaf_t)
        """
        if bigram_probs is None:
            bigram_probs = self.compute_empirical_bigram_distribution(num_samples)
        
        epsilon = 1e-10
        
        # Compute marginal P(leaf_t)
        marginal_probs = bigram_probs.sum(dim=1)  # Sum over next token
        
        # Compute conditional entropy
        conditional_entropy = 0.0
        
        for i in range(self.vocab_size):
            if marginal_probs[i] > epsilon:
                # Compute P(leaf_{t+1} | leaf_t = i)
                conditional_probs = bigram_probs[i] / marginal_probs[i]
                
                # Compute entropy of this conditional distribution
                log_probs = torch.log(conditional_probs + epsilon)
                entropy_i = -(conditional_probs * log_probs).sum().item()
                
                # Weight by marginal probability
                conditional_entropy += marginal_probs[i].item() * entropy_i
        
        return conditional_entropy
    
    def compute_empirical_marginal(self, bigram_probs=None, num_samples=100000):
        """
        Compute the marginal entropy H(leaf_{t+1}) from empirical bigram distribution.
        
        Args:
            bigram_probs: Pre-computed bigram probabilities (if None, will compute)
            num_samples: Number of samples for empirical estimation
            
        Returns:
            Marginal entropy H(leaf_{t+1})
        """
        if bigram_probs is None:
            bigram_probs = self.compute_empirical_bigram_distribution(num_samples)
        
        epsilon = 1e-10
        
        # Compute marginal P(leaf_{t+1})
        marginal_probs = bigram_probs.sum(dim=0)  # Sum over previous token
        
        # Compute entropy
        log_probs = torch.log(marginal_probs + epsilon)
        marginal_entropy = -(marginal_probs * log_probs).sum().item()
        
        return marginal_entropy
    
    def compute_all_empirical_statistics(self, num_samples=100000):
        """
        Compute all empirical statistics at once (more efficient).
        
        Args:
            num_samples: Number of samples for empirical estimation
            
        Returns:
            Dictionary with 'conditional_entropy', 'marginal_entropy', and 'bigram_probs'
        """
        print(f"\nComputing empirical statistics from {num_samples} samples...")
        
        # Compute bigram distribution once
        bigram_probs = self.compute_empirical_bigram_distribution(num_samples)
        
        # Compute both entropies from the same distribution
        conditional_entropy = self.compute_empirical_entropy(bigram_probs)
        marginal_entropy = self.compute_empirical_marginal(bigram_probs)
        
        # Store for later use
        self.empirical_entropy = conditional_entropy
        self.empirical_marginal = marginal_entropy
        
        print(f"Empirical conditional entropy: {conditional_entropy:.4f}")
        print(f"Empirical marginal entropy: {marginal_entropy:.4f}")
        
        return {
            'conditional_entropy': conditional_entropy,
            'marginal_entropy': marginal_entropy,
            'bigram_probs': bigram_probs
        }
