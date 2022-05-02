#!/usr/bin/env python
# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from email.mime import base
from math import prod, sqrt
from typing import Tuple
from mugnier.basis.dvr import SineDVR

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

    scale_factor = 1.0

    def __init__(self, sys_hamiltonian: Array, sys_op: Array, correlation: Correlation, dims: list[int]) -> None:

        self.h = sys_hamiltonian
        self.op = sys_op

        self.coefficients = correlation.coefficients
        self.conj_coefficents = correlation.conj_coefficents
        self.derivatives = correlation.derivatives
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
            _k = _end(k)
            ck = self.coefficients[k]
            cck = self.conj_coefficents[k]
            dk = self.derivatives[k]

            if self.scale_factor is None:
                fk = np.sqrt(np.real(ck + cck))
            else:
                fk = self.scale_factor
            print(f'{k}: fk={fk}; fk^(-1)={1.0 / fk}.')

            ops = [
                {
                    _k: dk * self.numberer(k)
                },
                {
                    _i: -1.0j * self.op,
                    _k: ck / fk * self.raiser(k) + fk * self.lower(k)
                },
                {
                    _j: 1.0j * self.op.conj(),
                    _k: cck / fk * self.raiser(k) + fk * self.lower(k)
                },
            ]
            ans.extend(ops)

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


class SineExtendedDensityTensor(CannonialModel):

    def __init__(self, rdo: Array, spaces: list[Tuple[float, float, int]]) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        bases = [SineDVR(*args) for args in spaces]
        dims = [b.n for b in bases]

        ends = [_end('i'), _end('j')] + [_end(k) for k in range(len(dims))]
        f = Singleton(ends)
        super().__init__(f, f.root)

        ext = zeros((prod(dims),))
        ext[0] = 1.0
        array = np.tensordot(rdo, ext, axes=0).reshape(shape + dims)
        for i, b in enumerate(bases):
            tfmat = b.fock2dvr_mat
            array = np.tensordot(array, tfmat, ([i+2], [1]))
            array = np.moveaxis(array, -1, i+2)

        self[self.root] = array
        return


class SineHierachy(Hierachy):

    def __init__(self, sys_hamiltonian: Array, sys_op: Array, correlation: Correlation,
                 spaces: list[Tuple[float, float, int]]) -> None:
        bases = [SineDVR(*args) for args in spaces]
        dims = [b.n for b in bases]
        self.bases = bases
        super().__init__(sys_hamiltonian, sys_op, correlation, dims)
        return

    def raiser(self, k: int):
        return self.bases[k].creation_mat

    def lower(self, k: int) -> Array:
        return self.bases[k].annihilation_mat

    def numberer(self, k: int) -> Array:
        return self.bases[k].numberer_mat
