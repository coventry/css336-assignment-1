"""Implementation of RMSNorm NN module

Response to problem `rmsnorm` of Assignment 1."""

import einx

import torch
from torch import nn, Tensor
from jaxtyping import Float


class RMSNorm(nn.Module):
    """RMS norm per equation (4) from the assignment

    Args:
        d_model: Hidden-dimension size
        eps: Small summand for numerical stability of denominator
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"hidden dim must be positive, got {d_model}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.d_model = d_model
        self.eps = eps
        gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)

    def forward(
        self, x: Float[Tensor, "*batch d_model"]
    ) -> Float[Tensor, "*batch d_model"]:
        "Returns [gᵢaᵢ/RMS(a)]ᵢ over last dim, formula (4) in the assignment"
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast to avoid overflow from squaring.
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
