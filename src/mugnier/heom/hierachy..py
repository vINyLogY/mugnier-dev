#!/usr/bin/env python
# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from math import prod

from mugnier.heom.bath import Correlation
from mugnier.libs.backend import Array, arange, eye, np, zeros
from mugnier.structure.frame import Singleton
from mugnier.structure.network import End, State
from mugnier.structure.operator import SumProdOp


class ExtendedDensityTensor(State):

    def __init__(self, k_max: int) -> None:
        ends = [End('H-i'), End('H-j')] + [End(f'H-{k}') for k in range(k_max)]
        f = Singleton(ends)
        super().__init__(f, f.root)
        return

    def initialize(self, rdo: Array, dims: list[int]) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        ext = zeros((prod(dims),))
        ext[0] = 1.0
        rho_n = np.tensordot(ext, rdo, axes=0).reshape(dims + shape)
        self[self.root] = rho_n

        return


class Hierachy(SumProdOp):

    def __init__(self, dims: list[int], sys_hamiltonian: Array, sys_op: Array, correlation: Correlation) -> None:
        self.k_max = len(dims)
        self.dims = dims
        self.h = sys_hamiltonian
        self.op = sys_op
        self.coefficients = correlation.coefficients
        self.conj_coefficents = correlation.conj_coefficents
        self.derivatives = correlation.derivatives

        super().__init__(self.op_list)
        return

    def op_list(self):
        _i = End('H-i')
        _j = End('H-j')
        ans = [
            {
                _i: -1.0j * self.h
            },
            {
                _j: 1.0j * self.h.conj()
            },
        ]

        for k in range(self.k_max):
            _k = End(f'H-{k}')
            ck = self.coefficients[k]
            cck = self.conj_coefficents[k]
            dk = self.derivatives[k]
            fk = 0.5

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
