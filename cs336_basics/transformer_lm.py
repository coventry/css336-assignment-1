"""Implementation of TransformerBlock.

Response to problem `transformer_block` of Assignment 1.
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
        self.layers = nn.ParameterList(  # num_layers transformer blocks
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
        x = self.token_embeddings(in_indices)
        for b in self.layers:
            x = b(x)
        return self.lm_head(self.ln_final(x))
