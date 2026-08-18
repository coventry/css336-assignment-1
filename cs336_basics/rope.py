"""Implementation of RoPE NN module.

Response to problem `rope` of Assignment 1.
"""

import einx

import torch
from torch import nn, Tensor

from jaxtyping import Float, Integer


class RoPE(nn.Module):
    """Rotate a query/key vector's coord pairs according to RoPE formula (8)

    Args:
        theta: base inverse rotation amount, capital Θ in §3.4.3 of assignment
        d_k: dimension of query/key vectors
        max_seq_len: Bound on seq len which can be input to forward
    """

    rotations: Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"d_k must be positive & even, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        thetas = torch.tensor(  # θᵢₖ in equation (8)
            [
                [
                    i / theta ** ((2 * k - 2) / d_k)
                    for k in range(1, d_k // 2 + 1)
                ]
                for i in range(max_seq_len)
            ],
            # Compute initially in 64-bit representation, to avoid rounding
            # errors.
            dtype=torch.float64,
            device=device,
        )
        cosines = torch.cos(thetas)
        sines = torch.sin(thetas)
        cosines = cosines.to(dtype or torch.get_default_dtype())
        sines = sines.to(dtype or torch.get_default_dtype())
        self.register_buffer(
            "rotations",  # rotations[i,k] = 2⨉2 θᵢₖ rotation matrix
            torch.stack(
                (
                    torch.stack((cosines, -sines), dim=-1),
                    torch.stack((sines, cosines), dim=-1),
                ),
                dim=-2,
            ),
            persistent=False,
        )

    def forward(
        self,
        inp: Float[Tensor, "*batch sequence_length d_k"],
        token_positions: Integer[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "*batch sequence_length d_k"]:
        """Rotate pairs in `inp` by rotations implied by token_positions."""
        if (latest := token_positions.max()) >= self.max_seq_len:
            raise ValueError(f"token position outside of max range: {latest}")
        if (earliest := token_positions.min()) < 0:
            raise ValueError(
                f"token_positions must be nonnegative, got {earliest}"
            )
        half_d_k = self.d_k // 2
        seqlen = token_positions.shape[-1]
        if (inp_seqlen := inp.shape[-2]) != seqlen:
            raise ValueError(
                f"Last dim of token_positions ({seqlen}) and second-last of "
                f"inp ({inp_seqlen}) must be equal."
            )
        qpairs = einx.id(  # pyright: ignore[reportPrivateImportUsage]
            "... seqlen (half_d_k pair) -> ... seqlen half_d_k pair",
            inp,
            seqlen=seqlen,
            half_d_k=half_d_k,
            pair=2,
        )
        rotations = self.rotations[token_positions]
        # Broadcast token_positions over any additional leading input
        # dimensions, such as batching/attention heads
        #
        # While it would be cleaner to broadcast token_positions, that leads to
        # copying all the rotation matrices during indexing.
        rotations = rotations.broadcast_to((*qpairs.shape[:-1], 2, 2))
        rotated = einx.dot(  # pyright: ignore[reportPrivateImportUsage]
            "... seqlen half_d_k row [col], ... seqlen half_d_k [col] -> "
            "... seqlen half_d_k row",
            rotations,
            qpairs,
            seqlen=seqlen,
            half_d_k=half_d_k,
            row=2,
            col=2,
        )
        return einx.id(  # pyright: ignore[reportPrivateImportUsage]
            "... seqlen half_d_k row -> ... seqlen (half_d_k row)",
            rotated,
            seqlen=seqlen,
            half_d_k=half_d_k,
            row=2,
        )
