# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from itertools import chain
from math import prod
from typing import Literal

from mugnier.basis.dvr import SineDVR

from mugnier.heom.bath import Correlation
from mugnier.libs.backend import MAX_EINSUM_AXES, Array, OptArray, arange, np, opt_einsum, opt_tensordot, optimize, zeros, dtype
from mugnier.libs.utils import huffman_tree, Optional
from mugnier.operator.spo import SumProdOp
from mugnier.state.frame import End, Frame, Node, Point
from mugnier.state.model import CannonialModel


class Hierachy(CannonialModel):

    @staticmethod
    def end( identifier: Literal['i', 'j'] | int) -> End:
        return End(f'[HEOM]{identifier}')

    @staticmethod
    def node(identifier: ...) -> Node:
        return Node(f'[HEOM]{identifier}')

    def __init__(self,
                 rdo: Array,
                 dims: list[int],
                 frame: Frame,
                 root: Node,
                 rank: int = 1,
                 decimation_rate: Optional[int] = None,
                 spaces: Optional[dict[int, tuple[float, float]]] = None) -> None:

        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]
        all_dims = shape + dims
        e_ends = [self.end('i'), self.end('j')]
        p_ends = [self.end(k) for k in range(len(dims))]
        all_ends = e_ends + p_ends

        assert root in frame.nodes
        assert frame.dual(root, 0)[0] == e_ends[0]
        assert frame.dual(root, 1)[0] == e_ends[1]
        assert set(all_ends) == frame.ends
        super().__init__(frame, root)

        dim_dct = {frame.dual(e, 0): d for d, e in zip(all_dims, all_ends)}
        axes = self.axes
        if decimation_rate is not None:
            for node in reversed(frame.node_visitor(root, 'BFS')):
                ax = axes[node]
                if ax is not None:
                    _ds = [dim_dct[node, i] for i in range(frame.order(node)) if i != ax]
                    dim_dct[frame.dual(node, ax)] = max(prod(_ds) // decimation_rate, rank)
        self.fill_eyes(dims=dim_dct, default_dim=rank)

        ext_shape = [k for i, k in enumerate(self.shape(root)) if i > 1]
        ext = zeros([prod(ext_shape)])
        ext[0] = 1.0
        array = np.tensordot(rdo, ext, axes=0).reshape(self.shape(root))
        self[root] = array

        # QFPE for defined k
        self.bases = dict()  # type: dict[End, SineDVR]
        tfmats = dict()  #type: dict[End, OptArray]
        if spaces is not None:
            for k, (x0, x1) in spaces.items():
                b = SineDVR(x0, x1, dims[k])
                tfmats[p_ends[k]] = optimize(b.fock2dvr_mat)
                self.bases[p_ends[k]] = b

            for _q, tfmat in tfmats.items():
                _p, _i = frame.dual(_q, 0)
                _a = opt_tensordot(self[_p], tfmat, ([_i], [1])).moveaxis(-1, _i)
                self.opt_update(_p, _a)

        # add terminators
        self.terminators = {}  # type: dict[Point, OptArray]
        for _k, d in zip(p_ends, dims):
            if _k in tfmats:
                tm = (tfmats[_k].mH)[0]
            else:
                tm = zeros([d])
                tm[0] = 1.0
                tm = optimize(tm)
            self.terminators[_k] = tm
        self.bfs_visitor = self.frame.node_visitor(root, method='BFS')
        return

    @staticmethod
    def terminate(tensor: OptArray, term_dict: dict[int, OptArray]):
        order = tensor.ndim
        n = len(term_dict)
        assert order + n - 1 < MAX_EINSUM_AXES

        ax_list = list(sorted(term_dict.keys(), key=(lambda ax: tensor.shape[ax])))
        vec_list = [term_dict[ax] for ax in ax_list]

        args = [(tensor, list(range(order)))]
        args.extend((vec_list[i], [ax_list[i]]) for i in range(n))
        ans_axes = [ax for ax in range(order) if ax not in ax_list]
        args.append((ans_axes,))

        args = list(chain(*args))
        ans = opt_einsum(*args)
        return ans

    def get_rdo(self) -> OptArray:
        axes = self.axes
        root = self.root
        terminators = self.terminators
        terminate = self.terminate
        near = self.frame.near_points

        # Iterative from leaves to root (not include)
        for p in reversed(self.bfs_visitor[1:]):
            term_dict = {i: terminators[q] for i, q in enumerate(near(p)) if i != axes[p]}
            terminators[p] = terminate(self[p], term_dict)

        # root node: 0 and 1 observed for i and j
        term_dict = {i: terminators[q] for i, q in enumerate(near(root)) if i > 1}
        # print(torch.norm(term_dict[2]))
        array = terminate(self[root], term_dict)
        dim = array.shape[0]
        return array.reshape((dim, dim, -1))[:, :, 0]


class NaiveHierachy(Hierachy):

    def __init__(
            self,
            rdo: Array,
            dims: list[int],
            spaces: Optional[dict[int, tuple[float, float]]] = None) -> None:
        ends = [self.end('i'), self.end('j')] + [self.end(k) for k in range(len(dims))]
        frame = Frame()
        root = Node(f'0')
        for e in ends:
            frame.add_link(root, e)
        super().__init__(rdo, dims, frame, root, spaces=spaces)
        return


class TreeHierachy(Hierachy):

    def __init__(
            self,
            rdo: Array,
            dims: list[int],
            n_ary=2,
            rank: int = 1,
            decimation_rate: Optional[int] = None,
            spaces: Optional[dict[int, tuple[float, float]]] = None) -> None:
        p_ends = [self.end(k) for k in range(len(dims))]
        frame = Frame()
        e_node = Node('Elec')
        frame.add_link(e_node, self.end('i'))
        frame.add_link(e_node, self.end('j'))

        class new_node(Node):
            _counter = 0

            def __init__(self) -> None:
                cls = type(self)
                super().__init__(f'T{cls._counter}')
                cls._counter += 1

        importances = list(dims)
        graph, p_node = huffman_tree(p_ends, new_node, importances=importances, n_ary=n_ary)
        frame.add_link(e_node, p_node)
        for n, children in graph.items():
            for child in children:
                frame.add_link(n, child)

        super().__init__(rdo,
                         dims,
                         frame,
                         e_node,
                         rank=rank,
                         decimation_rate=decimation_rate,
                         spaces=spaces)
        return


class TrainHierachy(Hierachy):

    def __init__(
            self,
            rdo: Array,
            dims: list[int],
            rev: bool = False,
            rank: int = 1,
            decimation_rate: Optional[int] = None,
            spaces: Optional[dict[int, tuple[float, float]]] = None) -> None:

        if rev:
            p_ends = [self.end(k) for k in reversed(range(len(dims)))]
        else:
            p_ends = [self.end(k) for k in range(len(dims))]

        frame = Frame()
        dof = len(dims)
        e_node = Node('Elec')
        frame.add_link(e_node, self.end('i'))
        frame.add_link(e_node, self.end('j'))
        p_nodes = [Node(f'{i}') for i in range(dof - 1)]
        if p_nodes:
            frame.add_link(e_node, p_nodes[0])
            for n in range(dof - 1):
                frame.add_link(p_nodes[n], p_ends[n])
                if n + 1 < dof - 1:
                    frame.add_link(p_nodes[n], p_nodes[n + 1])
            frame.add_link(p_nodes[-1], p_ends[-1])
        else:
            p_node = Node('0')
            frame.add_link(e_node, p_node)
            frame.add_link(p_node, p_ends[0])

        super().__init__(rdo,
                         dims,
                         frame,
                         e_node,
                         rank=rank,
                         decimation_rate=decimation_rate,
                         spaces=spaces)
        return


class HeomOp(SumProdOp):
    # ?
    scaling_factor = 2

    def __init__(self, hierachy: Hierachy, sys_hamiltonian: Array, sys_op: Array, correlation: Correlation,
                 dims: list[int]) -> None:
        self.bases = hierachy.bases
        self.end = hierachy.end
        self.h = sys_hamiltonian
        self.op = sys_op
        self.coefficients = correlation.coefficients
        self.conj_coefficents = correlation.conj_coefficents
        self.derivatives = correlation.derivatives
        self.dims = dims

        super().__init__(self.op_list)
        return

    @property
    def op_list(self) -> list[dict[End, Array]]:
        ans = [{self.end('i'): -1.0j * self.h}, {self.end('j'): 1.0j * self.h.conj()}]
        for k in range(len(self.dims)):
            ans.extend(self.bcf_term(k))
        return ans

    def bcf_term(self, k: int) -> list[dict[End, Array]]:
        _k = self.end(k)
        ck = self.coefficients[k]
        cck = self.conj_coefficents[k]
        dk = self.derivatives[k]
        if k in self.bases:
            fk = np.sqrt((ck.real + cck.real) / 2.0)
            raiser = self.bases[k].creation_mat
            lower = self.bases[k].annihilation_mat
            numberer = self.bases[k].numberer_mat
        else:
            fk = self.scaling_factor
            dim = self.dims[k]
            raiser = np.diag(np.sqrt(np.arange(1, dim, dtype=dtype)), k=-1)
            lower = np.diag(np.sqrt(np.arange(1, dim, dtype=dtype)), k=1)
            numberer = np.diag(arange(dim))

        ans = [{
            _k: dk * numberer
        }, {
            self.end('i'): -1.0j * self.op,
            _k: (ck / fk * raiser + fk * lower)
        }, {
            self.end('j'): 1.0j * self.op.conj(),
            _k: (cck / fk * raiser + fk * lower)
        }]
        return ans
