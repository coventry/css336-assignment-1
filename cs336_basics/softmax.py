"""Implementation of Softmax function.

Response to problem `softmax` of Assignment 1.
"""

from torch import Tensor

from jaxtyping import Float


def softmax(
    in_features: Float[Tensor, "..."], dim: int
) -> Float[Tensor, "..."]:
    "Return softmax value from eq. (10), over last dimension of in_features"
    # keepdim=True to broadcast the max-contracted dimension over in_features
    normalized = in_features - in_features.amax(dim=dim, keepdim=True)
    numerators = normalized.exp()
    denom = numerators.sum(dim=dim, keepdim=True)
    return numerators / denom
