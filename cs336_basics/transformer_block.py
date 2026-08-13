"""Implementation of TransformerBlock.

Response to problem `transformer_block` of Assignment 1.
"""

from torch import Tensor, nn, device, dtype

from jaxtyping import Float

from cs336_basics.multihead_self_attention import (
    MultiHeadSelfAttention,
)
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch sequence_length d_model"],
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))
