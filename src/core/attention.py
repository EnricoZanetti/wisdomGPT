"""
Originally derived from Andrej Karpathy's minGPT implementation.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    Implements a standard multi-head masked self-attention mechanism with
    an additional projection layer. This custom implementation provides
    flexibility and control over the attention mechanism.
    """

    def __init__(self, config):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "Embedding dimension must be divisible by the number of heads."

        # Projections for keys, queries, and values
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # Regularization layers
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Mask to enforce causality in the attention mechanism
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        """
        Computes the attention scores and performs the self-attention operation.
        Arguments:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            layer_past: Optional past layer information for speed optimization.
        """
        B, T, C = x.size()

        # Compute keys, queries, and values
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Perform scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        # Concatenate heads and apply the final projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Implements a cross-attention mechanism that attends between two input sequences.
    Adapts the self-attention mechanism to handle two distinct input tensors.
    """

    def __init__(self, config):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "Embedding dimension must be divisible by the number of heads."

        # Projections for keys, queries, and values
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # Regularization layers
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Mask to enforce causality in the attention mechanism
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        """
        Computes cross-attention between two input sequences.
        Arguments:
            x_kv: Input tensor used as keys and values (batch_size_k, seq_length_k, embedding_dim).
            x_q: Input tensor used as queries (batch_size_q, seq_length_q, embedding_dim).
        """
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # Compute keys, queries, and values
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2)
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2)
        v = (
            self.value(x_kv)
            .view(Bk, Tk, self.n_head, Ck // self.n_head)
            .transpose(1, 2)
        )

        # Perform scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :Tq, :Tk] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        # Concatenate heads and apply the final projection
        y = y.transpose(1, 2).contiguous().view(max(Bk, Bq), Tq, Cq)
        y = self.resid_drop(self.proj(y))
        return y
