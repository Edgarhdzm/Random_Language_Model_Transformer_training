import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .fcn import MLP


class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Heads.
    
    Args:
        input_dim: The dimension of input tokens.
        input_size: The (maximal) number of input tokens.
        num_heads: The number of heads.
        out_dim: The dimension of output tokens.
        dropout: The fraction of weights to zero via dropout.
        decoder: True for one-directional (causal) attention.
    """
    def __init__(
        self, input_dim, input_size, num_heads, out_dim, dropout=0, decoder=False
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "inner dim. must be multiple of num. heads"

        self.input_dim = input_dim
        self.input_size = input_size
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.decoder = decoder

        self.key = nn.Parameter(torch.empty(self.out_dim, self.input_dim))
        self.query = nn.Parameter(torch.empty(self.out_dim, self.input_dim))
        self.value = nn.Parameter(torch.empty(self.out_dim, self.input_dim))
        self.projection = nn.Parameter(torch.empty(self.out_dim, self.out_dim))

        if decoder:
            self.register_buffer("tril", torch.tril(torch.ones(self.input_size, self.input_size)))
        else:
            self.register_buffer("tril", torch.ones(self.input_size, self.input_size))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        std_kqv = 1.0 / math.sqrt(self.input_dim)
        std_proj = 1.0 / math.sqrt(self.out_dim)
        init.normal_(self.key, mean=0.0, std=std_kqv)
        init.normal_(self.query, mean=0.0, std=std_kqv)
        init.normal_(self.value, mean=0.0, std=std_kqv)
        init.normal_(self.projection, mean=0.0, std=std_proj)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).
        
        Returns:
            The output of a multi-head attention layer,
            of size (batch_size, input_size, out_dim)
        """
        B, T, C = x.size()

        k = F.linear(x, self.key, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.linear(x, self.query, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, self.value, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        weight = q @ k.transpose(-2, -1)
        weight = weight * (self.head_dim ** -0.5)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        out = (weight @ v).transpose(1, 2).reshape(B, T, -1)
        out = F.linear(out, self.projection, bias=None)

        return out


class DecoderBlock(nn.Module):
    """
    One Decoder Block.
    
    Args:
        embedding_dim: The dimension of the tokens (kept constant past embedding).
        input_size: The (maximal) number of input tokens.
        num_heads: The number of attention heads.
        dropout: The fraction of weights to zero via dropout.
        ffwd_size: Size of the MLP is ffwd_size*embedding_dim.        
    """
    def __init__(
        self, embedding_dim, input_size, num_heads, ffwd_size=4, dropout=0, decoder=False
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.attn = MultiHeadAttention(
            input_dim=embedding_dim,
            input_size=input_size,
            num_heads=num_heads,
            out_dim=embedding_dim,
            dropout=dropout,
            decoder=decoder,
        )
        self.ffwd = MLP(
            input_dim=embedding_dim,
            nn_dim=ffwd_size * embedding_dim,
            out_dim=embedding_dim,
            num_layers=1,
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x


class CLM(nn.Module):
    """
    Causal (decoder-only) Language Model.
    
    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        dropout: The fraction of weights to zero via dropout.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, ffwd_size, num_layers, dropout=0, share_emb=True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffwd_size = ffwd_size
        self.num_layers = num_layers
        self.share_emb = share_emb

        self.token_embedding_table = nn.Embedding(vocab_size, self.embedding_dim)
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                    ffwd_size=self.ffwd_size,
                    decoder=True,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size)
        if share_emb:
            self.lm_head.weight = self.token_embedding_table.weight
        
        # Initialize with smaller std to start at log(vocab_size)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with smaller std so initial loss is close to log(vocab_size).
        """
        # Small initialization for embeddings
        init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)
        init.normal_(self.position_embedding_table.weight, mean=0.0, std=0.02)
        
        # Very small initialization for output head (if not sharing embeddings)
        if not self.share_emb:
            init.normal_(self.lm_head.weight, mean=0.0, std=0.02 / math.sqrt(self.embedding_dim))
            if self.lm_head.bias is not None:
                init.zeros_(self.lm_head.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, num_tokens):
        for _ in range(num_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class MLM(nn.Module):
    """
    Causal (decoder-only) Language Model for next-token prediction.
    
    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        dropout: The fraction of weights to zero via dropout.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, ffwd_size, num_layers, dropout=0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffwd_size = ffwd_size
        self.num_layers = num_layers

        self.token_embedding_table = nn.Embedding(vocab_size + 1, self.embedding_dim)
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                    ffwd_size=self.ffwd_size,
                    decoder=True,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size)
        
        # Initialize with smaller std to start at log(vocab_size)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with smaller std so initial loss is close to log(vocab_size).
        
        For uniform distribution over vocab_size classes, cross-entropy loss = log(vocab_size).
        We scale down the output layer to achieve this.
        """
        # Small initialization for embeddings
        init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)
        init.normal_(self.position_embedding_table.weight, mean=0.0, std=0.02)
        
        # Very small initialization for output head to start near uniform distribution
        # Scale by 1/sqrt(embedding_dim) to keep logits small
        init.normal_(self.lm_head.weight, mean=0.0, std=0.02 / math.sqrt(self.embedding_dim))
        if self.lm_head.bias is not None:
            init.zeros_(self.lm_head.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
