"""Implementation of TransformerLM.

Response to problem `transformer_lm` of Assignment 1.
"""

import einx

import torch
from torch import Tensor, nn, device, dtype

from jaxtyping import Float, Integer

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Full transformer language model

    Args:
        vocab_size: Number of tokens in tokenizer vocabulary
        context_length: How far to look back in self-attention
        d_model: Hidden-dimension size
        num_layers: Number of transformer layers
        num_heads: Number of self-attention heads
        d_ff: Dimension of hidden layer in feedforward network
        rope_theta: Base angle for RoPE
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(  # num_layers transformer blocks
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, in_indices: Integer[Tensor, "batch_size sequence_length"]
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """Compute transformer outputs

        Args:
            in_indices: Token ids of sequences to process

        Returns:
            Logits over vocab tokens for each sequence position
        """
        x = self.token_embeddings(in_indices)
        for b in self.layers:
            x = b(x)
        return self.lm_head(self.ln_final(x))
