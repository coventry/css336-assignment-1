"""Implementation of Embedding NN module

Response to problem `embedding` of Assignment 1."""

import einx

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_
from jaxtyping import Integer, Float


class Embedding(nn.Module):
    """Table of learnable token embeddings

    Args:
        num_embeddings: Token-vocabulary size
        embedding_dim: Dimension of token embedding space

    Both arguments must be positive.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if num_embeddings <= 0:
            err = f"num_embeddings must be positive, got {num_embeddings}"
            raise ValueError(err)
        if embedding_dim <= 0:
            err = f"embedding_dim must be positive, got {embedding_dim}"
            raise ValueError(err)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        embeddings = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        trunc_normal_(embeddings, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(embeddings)

    def forward(
        self, token_ids: Integer[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch embedding_dim"]:
        """Return the embeddings for the given token_ids"""
        return einx.get_at(  # pyright: ignore[reportPrivateImportUsage]
            "[num_embeddings] embedding_dim, ...  -> ... embedding_dim",
            self.weight,
            token_ids,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
