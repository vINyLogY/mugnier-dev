#!/usr/bin/env python
# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from math import prod

from mugnier.heom.bath import Correlation
from mugnier.libs.backend import Array, arange, eye, np, zeros
from mugnier.operator.spo import SumProdOp
from mugnier.state.frame import End
from mugnier.state.model import CannonialModel
from mugnier.state.template import Singleton


def _end(identifier: ...) -> End:
    return End(f'HEOM-{identifier}')


class ExtendedDensityTensor(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int]) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        ends = [_end('i'), _end('j')] + [_end(k) for k in range(len(dims))]
        f = Singleton(ends)
        super().__init__(f, f.root)

        ext = zeros((prod(dims),))
        ext[0] = 1.0
        array = np.tensordot(rdo, ext, axes=0).reshape(shape + dims)
        self[self.root] = array
        return


class Hierachy(SumProdOp):

    def __init__(self, sys_hamiltonian: Array, sys_op: Array, correlation: Correlation, dims: list[int]) -> None:

        self.h = sys_hamiltonian
        self.op = sys_op

        self.coefficients = correlation.coefficients
        self.conj_coefficents = correlation.conj_coefficents
        self.derivatives = correlation.derivatives

        self.k_max = len(dims)
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

        for k in range(self.k_max):
            _k = _end(k)
            ck = self.coefficients[k]
            cck = self.conj_coefficents[k]
            dk = self.derivatives[k]
            fk = 1.0

            ops = [
                {
                    _k: dk * self._numberer(k)
                },
                {
                    _i: -1.0j * self.op,
                    _k: ck / fk * self._raiser(k) + fk * self._lower(k)
                },
                {
                    _j: 1.0j * self.op.conj(),
                    _k: cck / fk * self._raiser(k) + fk * self._lower(k)
                },
            ]
            ans.extend(ops)

        return ans

    def _raiser(self, k: int) -> Array:
        dim = self.dims[k]
        sqrt_n = np.diag(np.sqrt(arange(dim)))
        ans = sqrt_n @ eye(dim, dim, k=-1)
        return ans

    def _lower(self, k: int) -> Array:
        dim = self.dims[k]
        sqrt_n = np.diag(np.sqrt(arange(dim)))
        ans = eye(dim, dim, k=1) @ sqrt_n
        return ans

    def _numberer(self, k: int) -> Array:
        ans = np.diag(arange(self.dims[k]))
        return ans
