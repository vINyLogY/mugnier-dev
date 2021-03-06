# coding: utf-8
r"""Backend for accelerated array-operations.
"""

from math import log10
from typing import Callable, Optional
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
import torchdiffeq

from mugnier.libs.utils import count_calls

# import opt_einsum as oe

DOUBLE_PRECISION = True
FORCE_CPU = False
MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10
PI = np.pi


# Place to keep magic numbers
class Parameters:
    ode_rtol = 1.0e-5
    ode_atol = 1.0e-8
    svd_atol = 1.0e-8

    if FORCE_CPU:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    def __str__(self) -> str:
        string = f'ODE[{log10(self.ode_rtol):+.0f}]({log10(self.ode_atol):+.0f})'
        string += f' | SVD({log10(self.svd_atol):+.0f})'
        string += f' | Device:{self.device}'
        return string


parameters = Parameters()

# CPU settings
if DOUBLE_PRECISION:
    dtype = np.complex128
else:
    dtype = np.complex64
    parameters.ode_rtol = 1.0e-3
    parameters.ode_atol = 1.0e-6
    parameters.svd_atol = 1.0e-6
Array = NDArray[dtype]


def arange(n: int) -> Array:
    return np.arange(n, dtype=dtype)


def array(a: ArrayLike) -> Array:
    return np.array(a, dtype=dtype)


def zeros(shape: list[int]) -> Array:
    return np.zeros(shape, dtype=dtype)


def zeros_like(shape: list[int]) -> Array:
    return np.zeros_like(shape, dtype=dtype)


def ones(shape: list[int]) -> Array:
    return np.ones(shape, dtype=dtype)


def eye(m: int, n: int, k: int = 0) -> Array:
    return np.eye(m, n, k, dtype=dtype)


# GPU settings
if DOUBLE_PRECISION:
    opt_dtype = torch.complex128
else:
    opt_dtype = torch.complex64
OptArray = torch.Tensor


@torch.no_grad()
def opt_cat(tensors: list[OptArray]) -> OptArray:
    return torch.cat(tensors)


@torch.no_grad()
def opt_eye_like(a: OptArray) -> OptArray:
    m, n = a.shape
    return torch.eye(m, n, device=parameters.device)


@torch.no_grad()
def opt_split(tensors: OptArray, size_list: list[int]) -> list[OptArray]:
    return torch.split(tensors, size_list)


@torch.no_grad()
def optimize(array: ArrayLike) -> OptArray:
    ans = torch.tensor(array, dtype=opt_dtype, device=parameters.device)
    return ans


@torch.no_grad()
def opt_einsum(*args) -> OptArray:
    return torch.einsum(*args)


@torch.no_grad()
def opt_sum(array: OptArray, dim: int) -> OptArray:
    return torch.sum(array, dim=dim)


@torch.no_grad()
def opt_tensordot(a: OptArray, b: OptArray, axes: tuple[list[int],
                                                        list[int]]) -> OptArray:
    return torch.tensordot(a, b, dims=axes)


@torch.no_grad()
def opt_compressed_qr(a: OptArray,
                      rank: Optional[int] = None) -> tuple[OptArray, OptArray]:
    u, s, vh = torch.linalg.svd(a, full_matrices=False)

    # Calculate rank from atol
    tol = parameters.svd_atol
    total_error = 0.0
    for n, s_i in reversed(list(enumerate(s))):
        total_error += s_i
        if total_error > tol:
            rank = n + 1
            break
    # default
    if rank is None:
        rank = 1
    rank = max(3 * rank, 4)

    if rank is not None and rank <= len(s):
        s = s[:rank]
        u = u[:, :rank]
        vh = vh[:rank, :]
    s /= torch.sum(s)
    ss = s.diag().to(opt_dtype)

    return u, ss @ vh


@torch.no_grad()
def opt_svd(a: OptArray) -> tuple[OptArray, OptArray]:
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    reg = parameters.svd_atol * torch.ones_like(s)
    s = torch.maximum(s, reg)
    s /= torch.sum(s)
    # print(s)
    return u, s.to(opt_dtype), vh


# @torch.no_grad()
# def opt_svd(a: OptArray) -> tuple[OptArray, OptArray]:
#     u, s, vh = torch.linalg.svd(a, full_matrices=False)
#     total_error = 0.0
#     rank = 1
#     for n, s_i in reversed(list(enumerate(s))):
#         total_error += s_i
#         if total_error > PINV_TOL:
#             rank = n + 1
#             break
#     s = s[:rank].to(opt_dtype)
#     u = u[:, :rank]
#     vh = vh[:rank, :]

#     return u, s.to(opt_dtype), vh


@torch.no_grad()
def opt_regularized_qr(a: OptArray) -> tuple[OptArray, OptArray]:
    q, r = torch.linalg.qr(a, mode='reduced')
    reg = parameters.svd_atol * torch.eye(
        r.shape[0], r.shape[1], device=parameters.device, dtype=opt_dtype)
    r = torch.where(torch.abs(r) > torch.abs(reg), r, reg)
    return q, r


@torch.no_grad()
def odeint(func: Callable[[OptArray], OptArray],
           y0: OptArray,
           dt: float,
           method='dopri5') -> tuple[OptArray, int]:
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
    - Scipy compatable (slow):
        - 'BDF'
    """

    @count_calls
    def _func(_t, _y):
        """wrap a complex function to a 2D real function"""
        # print('t_eval = ', _t.cpu().numpy())
        y = func(_y[0] + 1.0j * _y[1])
        return torch.stack([y.real, y.imag])

    _y0 = torch.stack([y0.real, y0.imag])
    _t = torch.tensor([0.0, dt], device=parameters.device)
    if method == 'BDF':
        y1 = torchdiffeq.odeint(_func,
                                _y0,
                                _t,
                                method='scipy_solver',
                                options={'solver': 'BDF'})
    else:
        y1 = torchdiffeq.odeint(_func,
                                _y0,
                                _t,
                                method=method,
                                rtol=parameters.ode_rtol,
                                atol=parameters.ode_atol)
    return (y1[1][0] + 1.0j * y1[1][1]), _func.calls


@torch.no_grad()
def opt_pinv(a: OptArray) -> OptArray:
    return torch.linalg.pinv(a, atol=parameters.svd_atol)


@torch.no_grad()
def opt_inv(a: OptArray) -> OptArray:
    return torch.linalg.inv(a)


# @torch.no_grad()
# def opt_lstsq(a: OptArray, b: OptArray) -> OptArray:
#     """Find the x s.t. ax =b"""
#     ans = torch.linalg.lstsq(a.cpu(), b.cpu(), rcond=PINV_TOL)
#     return ans.solution.to(device)
