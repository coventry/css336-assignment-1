"""Implementation of SDPA function.

Response to problem `scaled_dot_product_attention` of Assignment 1.
"""

from math import sqrt

import einx

from torch import Tensor, inf

from jaxtyping import Float, Bool

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "*batch queries d_k"],
    K: Float[Tensor, "*batch keys d_k"],
    V: Float[Tensor, "*batch keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "*batch queries d_v"]:
    """Compute scaled dot-product attention.

    Args:
        Q: query vectors for this head
        K: key vectors for this head
        V: value vectors for this head
        mask: which Q/K pairs should contribute to the softmax.

    mask must allow at least one key per query-row

    Returns:
        Attention(Q,K,V) = softmax(mask(QKᵗ/√dₖ))V
    """
    d_k = Q.shape[-1]
    queries = Q.shape[-2]
    keys = K.shape[-2]
    unscaled_logits = einx.dot(  # pyright: ignore[reportPrivateImportUsage]
        "... queries [d_k], ... keys [d_k] -> ... queries keys",  # QKᵗ
        Q,
        K,
        queries=queries,
        keys=keys,
        d_k=d_k,
    )
    if mask is not None:
        assert mask.any(  # any fully masked query-row ⇒ NaN's during softmax
            dim=-1
        ).all(), "mask must allow at least one key per query."
        # Broadcast mask over any batches in the input
        mask = mask.broadcast_to(unscaled_logits.shape)
        # Force 0's in softmax where mask==False. (exp(-inf)=0). mask(QKᵗ)
        unscaled_logits = einx.where(  # pyright: ignore[reportPrivateImportUsage]
            f"... queries keys, ,... queries keys -> ... queries keys",
            # -----------------^-------------- Note empty argument
            # This corresponds to scalar `inf`.
            ~mask,
            -inf,
            unscaled_logits,
            queries=queries,
            keys=keys,
        )
    # softmax(mask(QKᵗ/√dₖ))V
    return einx.dot(  # pyright: ignore[reportPrivateImportUsage]
        "... queries [keys], ... [keys] d_v -> ... queries d_v",
        softmax(unscaled_logits / sqrt(d_k), -1),  # Prob. dist.'s over keys
        V,
        queries=queries,
        keys=keys,
        d_v=V.shape[-1],
    )
