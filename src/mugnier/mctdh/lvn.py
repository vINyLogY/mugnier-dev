# coding: utf-8
"""Generating the derivative of the total rho in SoP formalism.
"""

from inspect import trace
from itertools import chain, pairwise
from math import prod, sqrt
from optparse import Option
from typing import Optional
from mugnier.basis.dvr import SineDVR

from mugnier.heom.bath import Correlation, DiscreteVibration
from mugnier.libs.backend import Array, OptArray, arange, eye, np, opt_einsum, opt_tensordot, optimize, zeros
from mugnier.operator.spo import SumProdOp
from mugnier.state.frame import End, Frame, Node
from mugnier.state.model import CannonialModel
from mugnier.state.template import Singleton


def _end(identifier: ...) -> End:
    return End(f'LvN-{identifier}')


class SpinBosonDensityOperator(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int], beta: Optional[float] = None) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]
        dof = len(dims)

        ends = ([_end('i'), _end('j')] + [_end(f'i{k}') for k in range(dof)] + [_end(f'j{k}') for k in range(dof)])
        f = Singleton(ends)
        super().__init__(f, f.root)
        self.system_size = shape[0]
        self.bath_size = prod(dims)
        if beta is None:
            ext = zeros((prod(dims), prod(dims)))
            ext[0, 0] = 1.0
            array = np.tensordot(rdo, ext, axes=0).reshape(shape + dims + dims)
        else:
            array = rdo
            for dim in dims:
                d = [-beta * (n + 0.5) for n in range(dim)]
                rb = np.exp(d)
                rb = rb / np.sum(rb)
                array = np.tensordot(array, np.diag(rb), axes=0)
            array = array.reshape(shape + dims + dims)
            array = np.moveaxis(array, [3 + 2 * n for n in range(dof)], [2 + dof + n for n in range(dof)])
        self[self.root] = array
        return

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim_s = self.system_size
        dim_b = self.bath_size
        return opt_einsum(array.reshape((dim_s, dim_s, dim_b, dim_b)), [0, 1, 2, 2], [0, 1])


class TensorTrainDO(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int], beta: Optional[float] = None) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]
        dof = len(dims)

        f = Frame()
        e_node = Node(name='Elec')
        train = [e_node]
        f.add_link(e_node, _end('i'))
        f.add_link(e_node, _end('j'))
        p_ends = [_end(f'i{k}') for k in range(dof)] + [_end(f'j{k}') for k in range(dof)])
        for k, (i, j) in enumerate((p_ends)):
            i_node = Node(name=f'i{k}')
            j_node = Node(name=f'j{k}')
            f.add_link(train[-1], i_node)
            f.add_link(i_node, i)
            f.add_link(i_node, j_node)
            f.add_link(j_node, j)
            train.extend((i_node, j_node))

        super().__init__(f, e_node)
        self[self.root] = np.array(rdo)[:, :, np.newaxis]
        length = len(train)
        for n, (p, q) in enumerate(pairwise(train[1:])):
            dim = dims[n]
            d = [-beta * (n + 0.5) for n in range(dim)]
            rb = np.exp(d)
            rb = np.diag(np.sqrt(rb / np.sum(rb)))
            self[p] = rb[np.newaxis, :, :]
            if n != length:
                self[q] = rb[:, :, np.newaxis]
            else:
                self[q] = rb

        for _ in range(length - 1):
            self.unite_split_move(2)
        for _ in range(length - 1):
            self.unite_split_move(0)


        return

    def get_rdo(self):
        raise NotImplementedError



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
