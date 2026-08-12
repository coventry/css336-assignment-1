"""Implementation of SDPA function.

Response to problem `scaled_dot_product_attention` of Assignment 1.
"""

from math import sqrt

import einx

from torch import Tensor, inf

from jaxtyping import Float, Bool

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "*queries d_k"],
    K: Float[Tensor, "*keys d_k"],
    V: Float[Tensor, "*keys d_v"],
    mask: Bool[Tensor, "*queries keys"] | None = None,
) -> Float[Tensor, "*queries d_v"]:
    d_k = Q.shape[-1]
    queries = Q.shape[-2]
    keys = K.shape[-2]
    unscaled_logits = einx.dot(  # pyright: ignore[reportPrivateImportUsage]
        "... queries [d_k], ... keys [d_k] -> ... queries keys",
        Q,
        K,
        queries=queries,
        keys=keys,
        d_k=d_k,
    )
    if mask is not None:
        # Force 0's in softmax where mask==False. (exp(-inf)=0)
        unscaled_logits = einx.where(  # pyright: ignore[reportPrivateImportUsage]
            "... queries keys, ,... queries keys -> ... queries keys",
            # ----------------^-------------- Empty argument
            ~mask,
            -inf,
            unscaled_logits,
            queries=queries,
            keys=keys,
        )
    return einx.dot(  # pyright: ignore[reportPrivateImportUsage]
        "... queries [keys], ... [keys] d_v -> ... queries d_v",
        softmax(unscaled_logits / sqrt(d_k), -1),  # Prob. dist.'s over keys
        V,
        queries=queries,
        keys=keys,
        d_v=V.shape[-1],
    )
