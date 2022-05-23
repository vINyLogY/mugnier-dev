from functools import reduce
from math import prod
from typing import Callable, Iterable, Optional, Tuple

from mugnier.libs.backend import (Array, OptArray, array, eye, np, opt_qr, opt_tensordot, optimize, zeros)
from mugnier.state.frame import End, Frame, Node, Point
from mugnier.libs.utils import depths


def triangular(n_list):
    """A Generator yields the natural number in a triangular order.
        """
    length = len(n_list)
    prod_list = [1]
    for n in n_list:
        prod_list.append(prod_list[-1] * n)
    prod_list = prod_list

    def key(case):
        return sum(n * i for n, i in zip(prod_list, case))

    combinations = {0: [[0] * length]}
    for m in range(prod_list[-1]):
        if m not in combinations:
            permutation = [
                case[:j] + [case[j] + 1] + case[j + 1:]
                for case in combinations[m - 1]
                for j in range(length)
                if case[j] + 1 < n_list[j]
            ]
            combinations[m] = []
            for case in permutation:
                if case not in combinations[m]:
                    combinations[m].append(case)
        for case in combinations[m]:
            yield key(case)


class Model:
    """Network is a Frame with valuation for each node.
    """

    def __init__(self, frame: Frame) -> None:
        """
        Args:
            frame: Topology of the tensor network;
        """
        self.frame = frame
        self.ends = frame.ends
        self._dims = dict()  # type: dict[Tuple[Point, int], int]
        self._valuation = dict()  # type: dict[Node, OptArray]
        return

    def __contains__(self, item):
        return item in self.frame.nodes

    def items(self):
        return self._valuation.items()

    def shape(self, p: Point) -> list[Optional[int]]:
        assert p in self.frame.world

        if isinstance(p, End):
            dim = self._dims.get((p, 0))
            node_shape = [dim, dim]
        else:
            # isinstance(p, Node)
            node_shape = [self._dims.get((p, i)) for i in range(self.frame.order(p))]
        return node_shape

    def __getitem__(self, p: Node) -> OptArray:
        return self._valuation[p]

    def __setitem__(self, p: Node, array: Array) -> None:
        assert p in self.frame.nodes
        order = self.frame.order(p)
        assert array.ndim == order

        # Check confliction
        for i, dim in enumerate(array.shape):
            for pair in [(p, i), self.frame.dual(p, i)]:
                if pair in self._dims:
                    assert self._dims[pair] == dim
                else:
                    self._dims[pair] = dim

        self._valuation[p] = optimize(array)
        return

    def __delitem__(self, p: Node) -> None:
        for i in range(self.frame.order(p)):
            del self._dims[(p, i)]
        del self._valuation[p]
        return

    def opt_update(self, p: Node, array: OptArray) -> None:
        assert p in self._valuation
        self._valuation[p] = array
        self._dims.update({(p, i): dim for i, dim in enumerate(array.shape)})
        return

    def fill_zeros(self, dims: Optional[dict[Tuple[Node, int], int]] = None, default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the dimension for each Edge in dims (default is 1).
        """
        if dims is None:
            dims = dict()

        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        _dims = self._dims

        for p in self.frame.nodes:
            if p not in valuation:
                shape = [
                    _dims.get((p, i), dims.get((p, i), dims.get(dual(p, i), default_dim))) for i in range(order(p))
                ]
                self[p] = zeros(shape)

        return


class CannonialModel(Model):
    """
    State is a Network with Tree-shape and root.
    """

    def __init__(self, frame: Frame, root: Node) -> None:
        super().__init__(frame)

        self._root = root
        self._axes = None  # type: Optional[dict[Node, Optional[int]]]
        self._depths = None  # type: Optional[dict[Node, Optional[int]]]
        return

    @property
    def root(self) -> Node:
        return self._root

    @root.setter
    def root(self, value: Node) -> None:
        assert value in self.frame.nodes
        self._root = value
        self._axes = None
        self._depths = None
        return

    @property
    def axes(self) -> dict[Node, Optional[int]]:
        if self._axes is None:
            ans = {self.root: None}
            for m, _, n, j in self.frame.node_link_visitor(self.root):
                if m in ans and n not in ans:
                    ans[n] = j
            self._axes = ans

        return self._axes

    @property
    def depths(self) -> dict[Node, int]:
        if self._depths is None:
            ans = depths(self.root, self.frame.near_nodes)
            self._depths = ans

        return self._depths

    def fill_eyes(self, dims: Optional[dict[Tuple[Node, int], int]] = None, default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the each dimension tensors (default is 1).
        """
        if dims is None:
            dims = dict()
        nodes = self.frame.nodes
        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        saved_dims = self._dims
        axes = self.axes

        for p in nodes:
            if p not in valuation:
                ax = axes[p]
                shape = [
                    saved_dims.get((p, i), dims.get((p, i), dims.get(dual(p, i), default_dim)))
                    for i in range(order(p))
                ]
                if ax is None:
                    ans = zeros((prod(shape),))
                    ans[0] = 1.0
                    ans = ans.reshape(shape)
                else:
                    _m = shape.pop(ax)
                    _n = prod(shape)
                    # Naive
                    # ans = np.moveaxis(eye(_m, _n).reshape([_m] + shape), 0, ax)

                    # Triangular
                    ans = zeros([_m, _n])
                    for n, v_i in zip(triangular(shape), ans):
                        v_i[n] = 1.0
                    ans = np.moveaxis(ans.reshape([_m] + shape), 0, ax)


                self[p] = ans
        return

    def split_unite_move(self,
                         i: int,
                         op: Optional[Callable[[OptArray], OptArray]] = None,
                         rank: Optional[int] = None,
                         tol: Optional[float] = None) -> None:
        m = self.root
        assert i < self.frame.order(m)

        n, j = self.frame.dual(m, i)
        dim = self._dims[(m, i)]
        shape = self.shape(m)
        shape.pop(i)

        mat_m = self[m].moveaxis(i, -1).reshape((-1, dim))
        q, mid = opt_qr(mat_m, rank, tol)
        array_m = q.reshape(shape + [-1]).moveaxis(-1, i)
        self.opt_update(m, array_m)
        self.root = n

        if op is not None:
            mid = op(mid)

        array_n = opt_tensordot(mid, self[n], ([1], [j])).moveaxis(0, j)
        self.opt_update(n, array_n)
        return

    def unite_split_move(self,
                         i: int,
                         op: Optional[Callable[[OptArray], OptArray]] = None,
                         rank: Optional[int] = None,
                         tol: Optional[float] = None) -> None:
        m = self.root
        assert i < self.frame.order(m)

        n, j = self.frame.dual(m, i)
        shape_m = self.shape(m)
        shape_m.pop(i)
        shape_n = self.shape(n)
        shape_n.pop(j)

        mid = opt_tensordot(self[m], self[n], ([i], [j]))
        if op is not None:
            mid = op(mid)

        q, r = opt_qr(mid.reshape((prod(shape_m), prod(shape_n))), rank, tol)
        array_m = q.reshape(shape_m + [-1]).moveaxis(-1, i)
        self.opt_update(m, array_m)
        self.root = n

        array_n = r.reshape([-1] + shape_n).moveaxis(0, j)
        self.opt_update(n, array_n)
        return