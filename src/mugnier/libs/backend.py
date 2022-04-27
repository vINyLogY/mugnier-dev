# coding: utf-8
r"""Backend for accelerated array-operations.
"""

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

# import opt_einsum as oe

MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10
PI = np.pi

# CPU settings

dtype = np.complex128
Array = NDArray[dtype]


def arange(n: int) -> Array:
    return np.arange(n, dtype=dtype)


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
    device = 'cuda'
else:
    device = 'cpu'

# device = 'cpu'

opt_dtype = torch.complex128
OptArray = torch.Tensor


def optimize(array: ArrayLike) -> OptArray:
    ans = torch.tensor(array, dtype=opt_dtype, device=device).detach()
    return ans


def opt_einsum(*args) -> OptArray:
    return torch.einsum(*args).detach()


def opt_sum(array: OptArray, dim: int) -> OptArray:
    return torch.sum(array, dim=dim).detach()


# def opt_tensordot(a: OptArray, b: OptArray, axes: Tuple[list[int], list[int]]) -> OptArray:
#     return torch.tensordot(a, b, dims=axes).detach()
