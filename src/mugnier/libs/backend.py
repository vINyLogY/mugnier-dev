# coding: utf-8
r"""Backend for accelerated array-operations.
"""

from typing import Callable, Optional, Tuple
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
import torchdiffeq
import scipy.integrate

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


@torch.no_grad()
def optimize(array: ArrayLike) -> OptArray:
    ans = torch.tensor(array, dtype=opt_dtype, device=device)
    return ans


@torch.no_grad()
def opt_einsum(*args) -> OptArray:
    return torch.einsum(*args)


@torch.no_grad()
def opt_sum(array: OptArray, dim: int) -> OptArray:
    return torch.sum(array, dim=dim)


@torch.no_grad()
def opt_tensordot(a: OptArray, b: OptArray, axes: Tuple[list[int], list[int]]) -> OptArray:
    return torch.tensordot(a, b, dims=axes)


@torch.no_grad()
def opt_qr(a: OptArray, rank: Optional[int] = None, tol: Optional[float] = None) -> Tuple[OptArray, OptArray]:
    u, s, vh = torch.linalg.svd(a, full_matrices=False)

    # Calculate rank from atol
    if tol is not None:
        total_error = 0.0
        for n, s_i in reversed(list(enumerate(s))):
            total_error += s_i
            if total_error > tol:
                rank = n + 1
                break
        # default
        if rank is None:
            rank = 1

    if rank is not None and rank <= len(s):
        s = s[:rank]
        u = u[:, :rank]
        vh = vh[:rank, :]

    return u, s.diag() @ vh


@torch.no_grad()
def odeint(func: Callable[[OptArray], OptArray], y0: OptArray, dt: float, method='dopri5'):
    """Avaliable method:
    - Adaptive-step:
        - `dopri8` Runge-Kutta 7(8) of Dormand-Prince-Shampine
        - `dopri5` Runge-Kutta 4(5) of Dormand-Prince.
        - `bosh3` Runge-Kutta 2(3) of Bogacki-Shampine
        - `adaptive_heun` Runge-Kutta 1(2)
    - Fixed-step:
        - `euler` Euler method.
        - `midpoint` Midpoint method.
        - `rk4` Fourth-order Runge-Kutta with 3/8 rule.
        - `explicit_adams` Explicit Adams.
        - `implicit_adams` Implicit Adams.
    """

    def _func(_t, _y):
        """wrap a complex function to a 2D real function"""
        y = func(_y[0] + 1.0j * _y[1])
        return torch.stack([y.real, y.imag])

    _y0 = torch.stack([y0.real, y0.imag])
    _t = torch.tensor([0.0, dt], device=device)
    # y1 = torchdiffeq.odeint(_func, _y0, _t, method='scipy_solver', options={'solver': 'BDF'})
    y1 = torchdiffeq.odeint(_func, _y0, _t, method=method, atol=1.0e-11)
    return (y1[1][0] + 1.0j * y1[1][1])
