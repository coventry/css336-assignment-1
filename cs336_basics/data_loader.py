import numpy.typing as npt
import numpy as np

from torch import device, Tensor, from_numpy

r = np.random.default_rng(17)  # Repeatable batches


def get_batch(
    data: npt.NDArray, batch_size: int, context_length: int, device: device
) -> tuple[Tensor, Tensor]:
    starts = r.integers(len(data) - context_length, size=batch_size)
    offsets = np.arange(context_length)
    indices = starts[:, None] + offsets[None, :]
    inputs = from_numpy(data[indices])
    targets = from_numpy(data[indices + 1])
    return inputs.to(device=device), targets.to(device=device)
