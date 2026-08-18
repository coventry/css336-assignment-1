"""Implementation of cross-entropy.

Response to problem `cross_entropy` of Assignment 1."""

import einx

from jaxtyping import Float, Int

from numpy.testing import assert_allclose

from torch import Tensor, softmax, ones


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
    """Implementation of equ. (16)"""
    normalized = inputs - inputs.amax(dim=-1, keepdim=True)
    numerators = normalized.exp()
    log_denom = numerators.sum(dim=-1, keepdim=True).log()
    all_log_probs = normalized - log_denom
    assert_allclose(  # Verify that probs sum approx. to 1.
        all_log_probs.exp().sum(dim=-1), ones(inputs.shape[0]), rtol=1e-5
    )
    target_log_probs = einx.get_at(
        "... seq [vocab_log_probs], ... seq -> ... seq",
        all_log_probs,
        targets,
    )
    batch_size = targets.shape[0]
    return -target_log_probs.sum() / batch_size
