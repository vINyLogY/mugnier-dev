# coding: utf-8
r"""Data structure for topology of tensors in a network

"""

from __future__ import annotations
from email.policy import default

from itertools import pairwise
from math import prod
from typing import Literal, Optional, Tuple
from weakref import WeakValueDictionary

from mugnier.libs.backend import Array, OptArray, optimize, eye, moveaxis, zeros
from mugnier.libs.utils import depth_dict, iter_round_visitor, iter_visitor


class Point:
    __cache = WeakValueDictionary()  # type: WeakValueDictionary[str, Point]

    def __new__(cls, name: Optional[str] = None):
        if name is not None and name in cls.__cache:
            return cls.__cache[name]

        obj = object.__new__(cls)

        if name is not None:
            cls.__cache[name] = obj

        return obj

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = str(hex(id(self))) if name is None else str(name)
        return


class Node(Point):
    def __repr__(self) -> str:
        return f'({self.name})'


class End(Point):
    def __repr__(self) -> str:
        return f'<{self.name}>'


class Frame(object):
    def __init__(self):
        self._neighbor = dict()  # type: dict[Point, list[Point]]
        self._duality = dict()  # type: dict[Tuple[Point, int], Tuple[Point, int]]
        return

    @property
    def world(self) -> set[Point]:
        return set(self._neighbor.keys())

    @property
    def links(self) -> set[Tuple[Point, int, Point, int]]:
        return {(p, i, q, j) for (p, i), (q, j) in self._duality.items()}

    @property
    def nodes(self) -> set[Node]:
        return {p for p in self._neighbor.keys() if isinstance(p, Node)}

    @property
    def ends(self) -> set[End]:
        return {p for p in self._neighbor.keys() if isinstance(p, End)}

    def order(self, p: Node):
        return len(self._neighbor[p])

    def add_link(self, p: Point, q: Point) -> None:
        for _p in [p, q]:
            if _p not in self._neighbor:
                self._neighbor[_p] = []
            elif isinstance(_p, End):
                raise RuntimeError(f'End {_p} cannot link to more than one points.')

        i = len(self._neighbor[p])
        j = len(self._neighbor[q])
        self._duality[(p, i)] = (q, j)
        self._duality[(q, j)] = (p, i)

        self._neighbor[p].append(q)
        self._neighbor[q].append(p)
        return

    def near_points(self, key: Point) -> list[Point]:
        return list(self._neighbor[key])

    def near_nodes(self, key: Point) -> list[Node]:
        return [n for n in self._neighbor[key] if isinstance(n, Node)]

    def dual(self, p: Point, i: int) -> Tuple[Point, int]:
        return self._duality[(p, i)]

    def find_axes(self, p: Point, q: Point) -> Tuple[int, int]:
        i = self._neighbor[p].index(q)
        j = self._neighbor[q].index(p)
        return i, j

    def node_link_visitor(self, start: Node):
        nodes = [n for n in iter_round_visitor(start, self.near_nodes)]
        axes_list = [self.find_axes(n1, n2) for n1, n2 in pairwise(nodes)]
        return [(p, i, q, j) for (p, q), (i, j) in zip(pairwise(nodes), axes_list)]

    def node_visitor(self, start: Node, method: Literal['DFS', 'BFS'] = 'DFS'):
        return list(iter_visitor(start, self.near_nodes, method=method))

    def visitor(self, start: Point, method: Literal['DFS', 'BFS'] = 'DFS'):
        return list(iter_visitor(start, self.near_points, method=method))


class Network(object):
    """Network is a Frame with valuation for each node.
    """

    def __init__(self, frame: Frame) -> None:
        """
        Args:
            frame: Topology of the tensor network;
        """
        self.frame = frame
        self._dims = dict()  # type: dict[Tuple[Point, int], int]
        self._valuation = dict()  # type: dict[Node, OptArray]
        return

    def _claim_array(self, p: Node, array: Array) -> None:
        """Prepare the Network settings for a given array according to its shape
        """
        assert p in self.frame.nodes
        order = self.frame.order(p)
        assert order == array.ndim
        shape = array.shape

        for i in range(order):
            q, j = self.frame.dual(p, i)
            for pair in [(p, i), (q, j)]:
                if pair in self._dims:
                    assert self._dims[pair] == shape[i]
                else:
                    self._dims[pair] = shape[i]
        return

    def shape(self, p: Point) -> list[Optional[int]]:
        assert p in self.frame.world

        if isinstance(p, End):
            dim = self._dims.get((p, 0))
            node_shape = [dim, dim]
        else:
            for i in range(self.frame.order(p)):
                node_shape[i] = self._dims.get((p, i))
        return node_shape

    def __getitem__(self, node: Node) -> OptArray:
        return self._valuation[node]

    def __setitem__(self, node: Node, array: Array) -> None:
        self._claim_array(node, array)
        self._valuation[node] = optimize(array)
        return

    def __delitem__(self, node: Node) -> None:
        del self._valuation[node]
        return

    @property
    def ends(self):
        return self.frame.ends

    def fill_zeros(self, dims: Optional[dict[Tuple[Node, int], int]] = None, default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the dimension for each Edge in dims (default is 1).
        """
        if dims is None:
            dims = dict()

        nodes = self.frame.nodes
        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        saved_dims = self._dims

        for p in nodes:
            if p not in valuation:
                shape = [saved_dims.get((p, i), dims.get((p, i), dims.get(dual(p, i), default_dim)))
                         for i in range(order(p))]
                self[p] = zeros(shape)

        return


class State(Network):
    """
    State is a Network with Tree-shape and root.
    """

    def __init__(self, frame: Frame, root: Node) -> None:
        super().__init__(frame)

        self._root = root
        self._axes = None  # type: Optional[dict[Node, Optional[int]]]
        return

    @property
    def root(self) -> Node:
        return self._root

    @root.setter
    def root(self, value: Node) -> None:
        assert value in self.frame.nodes
        self._root = value
        self._axes = None
        return

    @property
    def axes(self) -> dict[Node, Optional[int]]:
        if self._axes is None:
            ans = {self.root: None}
            for m, _, n, j in self.frame.node_link_visitor(self.root):
                if m in ans and n not in ans:
                    ans[n] = j
            self._axes = ans
        else:
            ans = self._axes
        return ans

    @property
    def depth(self) -> dict[Point, int]:
        return depth_dict(self.root, self.frame.near_points)

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
                shape = [saved_dims.get((p, i), dims.get((p, i), dims.get(dual(p, i), default_dim)))
                         for i in range(order(p))]
                if ax is None:
                    ans = zeros((prod(shape), ))
                    ans[0] = 1.0
                    ans = ans.reshape(shape)
                else:
                    _m = shape.pop(ax)
                    _n = prod(shape)
                    ans = moveaxis(eye(_m, _n).reshape([_m] + shape), 0, ax)
                self[p] = ans
        return
