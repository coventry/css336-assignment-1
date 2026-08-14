"""Implementation of Linear NN module.

Response to problem `linear` of Assignment 1.
"""

from math import sqrt

import einx

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

from jaxtyping import Float


class Linear(nn.Module):
    """Linear multiplication of an input vector by a learnable weight matrix.

    Args:
        in_features: Expected size of input vectors
        out_features: Size of output vectors
        init_variance_scale: Optionally change σ² by this factor

    Both arguments must be positive.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_variance_scale: float = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features < 1:
            err = f"out_features must be positive, got {out_features}"
            raise ValueError(err)
        if init_variance_scale <= 0:
            raise ValueError(
                "init_variance_scale must be positive, got "
                f"{init_variance_scale}"
            )
        self.in_features = in_features
        self.out_features = out_features
        sigma = sqrt(2 * init_variance_scale / (in_features + out_features))
        bound = 3 * sigma
        init_weight = torch.empty(
            out_features, in_features, device=device, dtype=dtype
        )
        trunc_normal_(init_weight, mean=0, std=sigma, a=-bound, b=bound)
        self.weight = nn.Parameter(init_weight)

    def forward(
        self, inp: Float[Tensor, "*batch in_features"]
    ) -> Float[Tensor, "*batch out_features"]:
        """Multiply `inp` by self.weight, on the left."""
        return einx.dot(  # pyright: ignore[reportPrivateImportUsage]
            "out_features [in_features], ... [in_features] -> ... out_features",
            self.weight,
            inp,
            in_features=self.in_features,
            out_features=self.out_features,
        )
