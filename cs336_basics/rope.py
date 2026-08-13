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
            dtype=dtype,
            device=device,
        )
        cosines = torch.cos(thetas)
        sines = torch.sin(thetas)
        self.register_buffer(
            "rotations",  # rotations[i,k] = 2⨉2 θᵢₖ rotation matrix
            torch.stack(
                (
                    torch.stack((cosines, -sines), dim=-1),
                    torch.stack((sines, cosines), dim=-1),
                ),
                dim=-2,
            ),
        )

    def forward(
        self,
        inp: Float[Tensor, "*sequence_length d_k"],
        token_positions: Integer[Tensor, "*sequence_length"],
    ) -> Float[Tensor, "*sequence_length d_k"]:
        """Multiply `inp` by self.params, on the left."""
        if (latest := token_positions.max()) >= self.max_seq_len:
            raise ValueError(f"token position outside of max range: {latest}")
        half_d_k = self.d_k // 2
        seqlen = token_positions.shape[-1]
        assert inp.shape[-2] == seqlen
        qpairs = einx.id(  # pyright: ignore[reportPrivateImportUsage]
            "... seqlen (half_d_k pair) -> ... seqlen half_d_k pair",
            inp,
            seqlen=seqlen,
            half_d_k=half_d_k,
            pair=2,
        )
        tp_sig_prefix = ""
        if len(token_positions.shape) != 1:
            assert token_positions.shape[0] == 1
            assert (
                len(token_positions.shape) == 2
            ), f"length is {len(token_positions)}"
            token_positions = token_positions.squeeze(dim=0)
        rotated = einx.dot(  # pyright: ignore[reportPrivateImportUsage]
            "seqlen half_d_k row [col], ... seqlen half_d_k [col] -> "
            "... seqlen half_d_k row",
            self.rotations[token_positions],
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
