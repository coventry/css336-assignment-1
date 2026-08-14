"""Implementation of MultiHeadSelfAttention.

Response to problem `multi_head_self_attention` of Assignment 1.
"""

import einx

import torch
from torch import nn, device, dtype

from jaxtyping import Float, Integer

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


class MultiHeadSelfAttention(nn.Module):
    """Computes Multi-Head Self Attention according equ.'s 12-14 in assignment

    Args:
        d_model: Hidden-dimension size
        num_heads: Number of heads in multi-head
        max_seq_len: longest sequence over which attention will be computed
        theta: base rotation angle for rope

    If either of max_seq_len or theta is set, the other must be set, too.

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if theta is not None and theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_seq_len is None or theta is None:
            if (max_seq_len, theta) != (
                None,
                None,
            ):
                err = "Either set both max_seq_len & theta, or neither"
                raise ValueError(err)
        if d_model % num_heads != 0:
            raise ValueError(
                "num_heads must be a divisor of d_model, but"
                f"{num_heads} ∤ {d_model}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if theta is not None:
            if self.d_k % 2 != 0:
                raise ValueError(
                    "To use RoPE, attention-head internal dimension must be "
                    f"even, got {self.d_k}"
                )
        # Hidden-dim -> vertically stacked W_Q, W_K, W_V projections.
        # Assignment suggests this arrangement so that only a single matmul
        # occurs.
        self.attn_projections = Linear(
            d_model,
            3 * d_model,
            device=device,
            dtype=dtype,
            # Linear's initialization σ² assumes it's creating a single linear
            # function, but here we are abusing it to create three stacked
            # linear functions. Its default σ² for a (d_model, 3*d_model)
            # function is 1/(2*d_model) (see §3.3.1), but the required variance
            # for the actual (d_model, d_model) components is 1/d_model.
            # Therefore, we multiply the σ² initialization by 2 to compensate.
            init_variance_scale=2,
        )
        self.out_projection = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        if max_seq_len is not None:
            self.rope = RoPE(
                theta, self.d_k, max_seq_len, device=device, dtype=dtype
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "*batch sequence_length d_model"],
        token_positions: Integer[Tensor, "... sequence_length"] | None = None,
    ) -> Float[Tensor, " *batch sequence_length d_model"]:
        """Return equ. (14) from assignment.

        Args:
            x: hidden-dimension input to self-attention
            token_positions: indices in the sequence from which the x's come.

        token_positions must be strictly increasing.

        """
        # Compute W_Qx, W_Kx, W_Vx (Definitions below (14))
        q, k, v = einx.id(  # pyright: ignore[reportPrivateImportUsage]
            "... (num_proj num_heads d_k) -> num_proj num_heads ... d_k",
            self.attn_projections(x),  # Stacked q/k/v
            num_proj=3,  # I.e., q/k/v end up stacked on first tensor ordinate
            num_heads=self.num_heads,  # I.e., heads stacked on second ordinate
            d_k=self.d_k,  # per-head vectors in last ordinate
        )
        sequence_length = x.shape[-2]
        if self.rope is not None:  # Compute rope embeddings
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            else:
                # token_positions should strictly increase
                if torch.any(
                    token_positions[..., 1:] <= token_positions[..., :-1]
                ):
                    raise ValueError("Token positions must strictly increase.")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        # Since token positions are sorted, the lower-triangular mask is causal
        mask = torch.tril(  # Causal mask
            torch.empty(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ).fill_(True)
        )
        # Restack per-head outputs into last ordinate
        attention = einx.id(  # pyright: ignore[reportPrivateImportUsage]
            "num_heads ... d_v -> ... (num_heads d_v)",  # Equ. (12)
            scaled_dot_product_attention(q, k, v, mask),  # Equ. (13)
            num_heads=self.num_heads,
            d_v=self.d_k,
        )
        # Equ. (14)
        return self.out_projection(attention)  # Equ. (14)
