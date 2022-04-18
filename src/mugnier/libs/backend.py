# coding: utf-8
r"""Backend for accelerated array-operations.
"""
from functools import partial
from turtle import shape
from typing import Optional
from numpy import moveaxis

import torch

if torch.cuda.is_available():
    device = "cuda" 
else:
    device = 'cpu'

dtype = torch.complex64
Array = torch.Tensor
tensordot = torch.tensordot
reshape = torch.reshape
moveaxis = torch.movedim


def asarray(array):
    return torch.tensor(array, dtype=dtype, device=device).detach()


def zeros(shape):
    return torch.zeros(shape, dtype=dtype, device=device).detach()
