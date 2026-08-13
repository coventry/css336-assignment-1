"""Implementation of RMSNorm NN module

Response to problem `rmsnorm` of Assignment 1."""

import einx

import torch
from torch import nn, Tensor
from jaxtyping import Float


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)

    def forward(
        self, x: Float[Tensor, "*batch d_model"]
    ) -> Float[Tensor, "*batch d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast to avoid oveflow from squaring.
        length_sq = einx.dot(  # pyright: ignore[reportPrivateImportUsage]
            "... d_model, ... d_model -> ...", x, x, d_model=self.d_model
        )
        rms = torch.sqrt(length_sq / self.d_model + self.eps)
        gained = einx.multiply(  # pyright: ignore[reportPrivateImportUsage]
            "... d_model, d_model -> ... d_model",
            x,
            self.gain,
            d_model=self.d_model,
        )
        return einx.divide(  # pyright: ignore[reportPrivateImportUsage]
            "... d_model, ... -> ... d_model", gained, rms, d_model=self.d_model
        ).to(in_dtype)
