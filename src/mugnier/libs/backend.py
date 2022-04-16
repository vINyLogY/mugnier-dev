# coding: utf-8
r"""Backend for accelerated array-operations.
"""
from functools import partial

import torch

if torch.cuda.is_available():
    device = "cuda" 
else:
    device = 'cpu'

dtype = torch.complex64
Array = torch.Tensor
asarray = partial(torch.tensor, dtype=dtype, device=device)
zeros = partial(torch.zeros, dtype=dtype, device=device)
