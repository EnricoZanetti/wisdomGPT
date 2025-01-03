"""
GPT Model:
- The initial stem combines token encoding and positional encoding.
- The core architecture consists of a sequence of Transformer blocks:
  - Each Transformer block includes a self-attention module and a feed-forward module (MLP).
  - These modules interact through residual connections, inspired by ResNets.
- The final output is produced via a linear projection followed by a Softmax classifier.

Originally derived from Andrej Karpathy's minGPT.

Stanford XCS224N: Homework 5
Authors:
- John Hewitt <johnhew@stanford.edu>
- Ansh Khurana <anshk@stanford.edu>
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import CausalSelfAttention, CausalCrossAttention


class GPTConfig:
    """Base GPT configuration with parameters common to all GPT variants."""

    embd_pdrop = 0.1  # Dropout probability for embeddings
    resid_pdrop = 0.1  # Dropout probability for residuals
    attn_pdrop = 0.1  # Dropout probability for attention
    perceiver = False  # Flag to enable Perceiver variant
    bottleneck_dim = None  # Dimension of the Perceiver bottleneck, if used

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """Configuration for a GPT-1 like model with ~125M parameters."""

    n_layer = 12  # Number of Transformer layers
    n_head = 12  # Number of attention heads
    n_embd = 768  # Embedding size


class Block(nn.Module):
    """Defines a standard Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor after self-attention and feed-forward layers.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DownProjectBlock(nn.Module):
    """Transformer block for down-projection in the Perceiver model."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CausalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        # Basis vectors for the bottleneck
        self.C = nn.Parameter(torch.Tensor(1, config.bottleneck_dim, config.n_embd))
        nn.init.xavier_uniform_(self.C)

    def forward(self, x_input):
        """
        Forward pass through the down-projection block.

        Args:
            x_input: Input tensor.

        Returns:
            Transformed tensor with reduced dimensionality via cross-attention.
        """
        normalized_x_input = self.ln1(x_input)
        normalized_C = self.ln1(self.C)
        cross_attn_output = self.cross_attn(normalized_x_input, normalized_C)
        residual_connection = cross_attn_output + x_input
        cross_attn_output = self.ln2(residual_connection)
        output = self.mlp(cross_attn_output)
        return output


class UpProjectBlock(nn.Module):
    """Transformer block for up-projection in the Perceiver model."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CausalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, y, x_input):
        """
        Forward pass through the up-projection block.

        Args:
            y: Input tensor from the previous layer.
            x_input: Original input tensor for cross-attention.

        Returns:
            Transformed tensor with increased dimensionality via cross-attention.
        """
        normalized_y = self.ln1(y)
        normalized_x = self.ln1(x_input)
        cross_attn_output = self.cross_attn(normalized_y, normalized_x)
        residual_connection = cross_attn_output + y
        cross_attn_output = self.ln2(residual_connection)
        output = self.mlp(cross_attn_output)
        return output


class GPT(nn.Module):
    """Defines the full GPT model with configurable Perceiver variant."""

    def __init__(self, config):
        super().__init__()

        # Input embedding and positional encoding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer layers
        self.perceiver = config.perceiver
        if self.perceiver:
            input_block_size = config.block_size
            self.down_block = DownProjectBlock(config)
            config.block_size = config.bottleneck_dim
            self.blocks = nn.Sequential(
                *[Block(config) for _ in range(config.n_layer - 2)]
            )
            config.block_size = input_block_size
            self.up_block = UpProjectBlock(config)
        else:
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # Final layer normalization and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        """
        Initializes the weights of the model.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        """
        Returns the maximum sequence length supported by the model.
        """
        return self.block_size

    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.

        Args:
            idx: Input tensor of token indices.
            targets: Target tensor for loss calculation (optional).

        Returns:
            logits: Logits from the output layer.
            loss: Cross-entropy loss (if targets are provided).
        """
        b, t = idx.size()
        assert t <= self.block_size, f"Block size exceeded: {t} > {self.block_size}"

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x_input = self.drop(token_embeddings + position_embeddings)

        if self.perceiver:
            x = self.down_block(x_input)
        else:
            x = x_input

        x = self.blocks(x)

        if self.perceiver:
            x = self.up_block(x, x_input)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0
            )

        return logits, loss
