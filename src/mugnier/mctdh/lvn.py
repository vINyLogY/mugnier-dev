# coding: utf-8
"""Generating the derivative of the total rho in SoP formalism.
"""

from itertools import chain
from math import prod, sqrt
from optparse import Option
from typing import Optional, Tuple
from mugnier.basis.dvr import SineDVR

from mugnier.heom.bath import Correlation, DiscreteVibration
from mugnier.libs.backend import Array, OptArray, arange, eye, np, opt_einsum, opt_tensordot, optimize, zeros
from mugnier.operator.spo import SumProdOp
from mugnier.state.frame import End
from mugnier.state.model import CannonialModel
from mugnier.state.template import Singleton


def _end(identifier: ...) -> End:
    return End(f'LvN-{identifier}')


class SpinBosonDensityOperator(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int], beta: Optional[float] = None) -> None:
        if beta is not None:
            raise NotImplementedError

        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        ends = ([_end('i'), _end('j')] + [_end(f'i{k}') for k in range(len(dims))] +
                [_end(f'j{k}') for k in range(len(dims))])
        f = Singleton(ends)
        super().__init__(f, f.root)

        # ZT case
        ext = zeros((prod(dims), prod(dims)))
        ext[0, 0] = 1.0
        array = np.tensordot(rdo, ext, axes=0).reshape(shape + dims + dims)
        self[self.root] = array
        self.system_size = shape[0]
        self.bath_size = prod(dims)
        return

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim_s = self.system_size
        dim_b = self.bath_size
        return opt_einsum(array.reshape((dim_s, dim_s, dim_b, dim_b)), [0, 1, 2, 2], [0, 1])


class SpinBosonLvN(SumProdOp):

    def __init__(self, sys_hamiltonian: Array, sys_op: Array, bathes: list[DiscreteVibration],
                 dims: list[int]) -> None:

        self.h = sys_hamiltonian
        self.op = sys_op

        self.gs = [b.g for b in bathes]
        self.w0s = [b.w0 for b in bathes]
        self.dims = dims

        super().__init__(self.op_list)
        return

    @property
    def op_list(self):
        _i = _end('i')
        _j = _end('j')
        ans = [
            {
                _i: -1.0j * self.h
            },
            {
                _j: 1.0j * self.h.conj()
            },
        ]

        for k in range(len(self.dims)):
            _ik = _end(f'i{k}')
            _jk = _end(f'j{k}')
            g = self.gs[k]
            w0 = self.w0s[k]

            opk = [
                {
                    _ik: -1.0j * w0 * self.numberer(k)
                },
                {
                    _jk: 1.0j * w0 * self.numberer(k)
                },
                {
                    _i: -1.0j * self.op,
                    _ik: g * (self.raiser(k) + self.lower(k))
                },
                {
                    _j: 1.0j * self.op.conj(),
                    _jk: g * (self.raiser(k) + self.lower(k))
                },
            ]
            ans.extend(opk)

        return ans

    def raiser(self, k: int) -> Array:
        dim = self.dims[k]
        return np.diag(np.sqrt(arange(dim))) @ eye(dim, dim, k=-1)

    def lower(self, k: int) -> Array:
        dim = self.dims[k]
        return eye(dim, dim, k=1) @ np.diag(np.sqrt(arange(dim)))

    def numberer(self, k: int) -> Array:
        dim = self.dims[k]
        return np.diag(arange(dim))
