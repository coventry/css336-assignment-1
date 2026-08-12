"""Implementation of SwiGLU NN module.

Response to problem `positionwise_feedforward` of Assignment 1.
"""

from jaxtyping import Float

import torch
from torch import nn, Tensor

from cs336_basics.linear import Linear


def silu(x: Float[Tensor, "*args"]) -> Float[Tensor, "*args"]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if min(d_model, d_ff) < 1:
            raise ValueError(
                f"(d_model, d_ff) must both be positive, got {(d_model, d_ff)}"
            )
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        "Returns W₂(SiLU(W₁x)⊙W₃), formula 7 in the assignment."
        assert x.shape[-1] == self.d_model
        return self.w2(silu(self.w1(x)) * self.w3(x))
