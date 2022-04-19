# coding: utf-8
r"""Backend for accelerated array-operations.
"""
from functools import partial
from turtle import shape
from typing import Optional
from numpy.typing import ArrayLike
from sympy import arg

import torch

MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10

if torch.cuda.is_available():
    device = "cuda" 
else:
    device = 'cpu'

dtype = torch.complex64
Array = torch.Tensor
tensordot = torch.tensordot
reshape = torch.reshape
moveaxis = torch.movedim


def as_array(array: ArrayLike) -> Array:
    return torch.tensor(array, dtype=dtype, device=device).detach()


def zeros(shape: list[int]) -> Array:
    return torch.zeros(shape, dtype=dtype, device=device).detach()


def eye(m: int, n: int) -> Array:
    return torch.eye(m, n, dtype=dtype, device=device).detach()


def stack(tensors: list[Array]) -> Array:
    return torch.stack(tensors).detach()


def einsum(*args) -> Array:
    return torch.einsum(*args).detach()
