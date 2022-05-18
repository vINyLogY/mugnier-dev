# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from math import prod, sqrt
from typing import Tuple
from mugnier.basis.dvr import SineDVR

from mugnier.heom.bath import Correlation
from mugnier.libs.backend import Array, OptArray, arange, eye, np, opt_tensordot, optimize, zeros
from mugnier.operator.spo import SumProdOp
from mugnier.state.frame import End, Frame, Node
from mugnier.state.model import CannonialModel
from mugnier.state.template import Singleton, TensorTrain


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

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim = array.shape[0]
        return array.reshape((dim, dim, -1))[:, :, 0]


class Layer3EDT(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int], rank: int = 1) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        ends = [_end(k) for k in range(len(dims))]
        f = Frame()
        dof = len(dims)
        e_node = Node('elec')
        r_node = Node('root')

        spf_nodes = [Node(f'{i}') for i in range(dof)]
        f.add_link(e_node, _end('i'))
        f.add_link(e_node, _end('j'))

        f.add_link(e_node, r_node)
        for n in range(dof):
            f.add_link(r_node, spf_nodes[n])
            f.add_link(spf_nodes[n], ends[n])

        super().__init__(f, e_node)

        _r = prod(rdo.shape)
        ext = zeros((_r,))
        ext[0] = 1.0
        array = np.tensordot(rdo, ext, axes=0)
        self[self.root] = array.reshape(list(rdo.shape) + [_r])
        dim_dct = {f.dual(e, 0): dims[i] for i, e in enumerate(ends)}
        dim_dct.update({(r_node, 0): _r})
        # if rank > prod(shape):
        #     dim_dct[e_node, 2] = prod(shape)
        self.fill_eyes(dims=dim_dct, default_dim=rank)
        return

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim = array.shape[0]
        return array.reshape((dim, dim, -1))[:, :, 0]


class Layer2EDT(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int], rank: int = 1) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        ends = [_end(k) for k in range(len(dims))]
        f = Frame()
        dof = len(dims)
        r_node = Node('root')
        spf_nodes = [Node(f'{i}') for i in range(dof)]
        f.add_link(r_node, _end('i'))
        f.add_link(r_node, _end('j'))
        for n in range(dof):
            f.add_link(r_node, spf_nodes[n])
            f.add_link(spf_nodes[n], ends[n])

        super().__init__(f, r_node)

        ext = zeros((rank**dof,))
        ext[0] = 1.0
        array = np.tensordot(rdo, ext, axes=0)
        self[self.root] = array.reshape(list(rdo.shape) + [rank] * dof)
        dim_dct = {f.dual(e, 0): dims[i] for i, e in enumerate(ends)}
        # if rank > prod(shape):
        #     dim_dct[e_node, 2] = prod(shape)
        self.fill_eyes(dims=dim_dct, default_dim=rank)
        return

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim = array.shape[0]
        return array.reshape((dim, dim, -1))[:, :, 0]


class TensorTrainEDT(CannonialModel):

    def __init__(self, rdo: Array, dims: list[int], rank: int = 1) -> None:
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]

        ends = [_end(k) for k in range(len(dims))]
        f = Frame()
        dof = len(dims)
        e_node = Node('Elec')
        p_nodes = [Node(f'{i}') for i in range(dof - 1)]
        f.add_link(e_node, _end('i'))
        f.add_link(e_node, _end('j'))
        f.add_link(e_node, p_nodes[0])
        if dof > 1:
            for n in range(dof - 2):
                f.add_link(p_nodes[n], p_nodes[n + 1])
        for n in range(dof - 1):
            f.add_link(p_nodes[n], ends[n])
        f.add_link(p_nodes[-1], ends[-1])

        super().__init__(f, e_node)

        ext = zeros((rank,))
        ext[0] = 1.0
        array = np.tensordot(rdo, ext, axes=0)
        self[self.root] = array
        dim_dct = {f.dual(e, 0): dims[i] for i, e in enumerate(ends)}
        # if rank > prod(shape):
        #     dim_dct[e_node, 2] = prod(shape)
        self.fill_eyes(dims=dim_dct, default_dim=rank)
        return

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim = array.shape[0]
        return array.reshape((dim, dim, -1))[:, :, 0]


class Hierachy(SumProdOp):
    scaling_factor = 1.0

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
            if self.scaling_factor is None:
                fk = np.sqrt(np.real(ck + cck))
            else:
                fk = self.scaling_factor
            # print(f'f_{k} = {fk}')

            opk = [
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


class SineExtendedDensityTensor(ExtendedDensityTensor):
    scale_factor = 1.0

    def __init__(self, rdo: Array, spaces: list[Tuple[float, float, int]]) -> None:
        bases = [SineDVR(*args) for args in spaces]
        dims = [b.n for b in bases]
        tfmats = [optimize(b.fock2dvr_mat) for b in bases]
        super().__init__(rdo, dims)

        array = self[self.root]
        for i, tfmat in enumerate(tfmats):
            array = opt_tensordot(array, tfmat, ([i + 2], [1])).moveaxis(-1, i + 2)
        self.opt_update(self.root, array)

        self.tfmats = tfmats
        return

    def get_rdo(self) -> OptArray:
        array = self[self.root]
        dim = array.shape[0]
        for i, tfmat in enumerate(self.tfmats):
            array = opt_tensordot(array, tfmat.T, ([i + 2], [1])).moveaxis(-1, i + 2)
        return array.reshape((dim, dim, -1))[:, :, 0]


class SineHierachy(Hierachy):

    def __init__(self, sys_hamiltonian: Array, sys_op: Array, correlation: Correlation,
                 spaces: list[Tuple[float, float, int]]) -> None:
        bases = [SineDVR(*args) for args in spaces]
        dims = [b.n for b in bases]
        self.bases = bases
        super().__init__(sys_hamiltonian, sys_op, correlation, dims)
        return

    def raiser(self, k: int) -> Array:
        return self.bases[k].creation_mat

    def lower(self, k: int) -> Array:
        return self.bases[k].annihilation_mat

    def numberer(self, k: int) -> Array:
        return self.bases[k].numberer_mat
