# coding: utf-8
r"""Backend for accelerated array-operations.
"""

from numpy.typing import ArrayLike, NDArray
from numpy import stack, moveaxis, reshape

import torch
import numpy as np

MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10

# CPU settings

dtype = np.complex64
Array = NDArray[dtype]


def array(a: ArrayLike) -> Array:
    return np.array(a, dtype=dtype)


def zeros(shape: list[int]) -> Array:
    return np.zeros(shape, dtype=dtype)


def ones(shape: list[int]) -> Array:
    return np.ones(shape, dtype=dtype)


def eye(m: int, n: int, k: int = 0) -> Array:
    return np.eye(m, n, k, dtype=dtype)


# GPU settings

if torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'

opt_dtype = torch.complex64
OptArray = torch.Tensor

opt_reshape = torch.reshape
opt_moveaxis = torch.movedim


def optimize(array: ArrayLike) -> OptArray:
    return torch.tensor(array, dtype=opt_dtype, device=device).detach()


def opt_einsum(*args) -> OptArray:
    return torch.einsum(*args).detach()


def opt_sum(array: OptArray, dim: int) -> OptArray:
    return torch.sum(array, dim=dim).detach()




def opt_exp(array: OptArray) -> OptArray:
    return torch.matrix_exp(array).detach()
