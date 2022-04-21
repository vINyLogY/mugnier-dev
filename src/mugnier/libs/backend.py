# coding: utf-8
r"""Backend for accelerated array-operations.
"""

from numpy.typing import ArrayLike, NDArray
from numpy import array, stack, moveaxis, reshape

import torch
import numpy as np

MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10

if torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'

dtype = np.complex64
dtype_gpu = torch.complex64
Array = NDArray[dtype]
ArrayGPU = torch.Tensor

# CPU arrays


def zeros(shape: list[int]) -> Array:
    return np.zeros(shape, dtype=dtype)


def ones(shape: list[int]) -> Array:
    return np.ones(shape, dtype=dtype)


def eye(m: int, n: int, k: int = 0) -> Array:
    return np.eye(m, n, k, dtype=dtype)


reshape_gpu = torch.reshape
moveaxis_gpu = torch.movedim


def to_gpu(array: ArrayLike) -> Array:
    return torch.tensor(array, dtype=dtype_gpu, device=device).detach()


def einsum_gpu(*args) -> Array:
    return torch.einsum(*args).detach()
