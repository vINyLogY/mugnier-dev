# coding: utf-8
r"""Data structure for topology of tensors in a network

"""

from __future__ import annotations
from collections import OrderedDict
from functools import partial
from itertools import pairwise

from typing import Iterable, Literal, Optional

from matplotlib.pyplot import axes

from mugnier.libs.utils import T, Cached, depth_dict, iter_visitor, iter_round_visitor, lazyproperty
from mugnier.libs.backend import asarray, Array, zeros


class Node(metaclass=Cached):

    def __init__(self, name: Optional[str] = None) -> None:
        """Node monad."""
        self.name = str(hex(id(self))) if name is None else str(name)
        return

    def __repr__(self) -> str:
        return f'({self.name})'


class Edge(metaclass=Cached):

    def __init__(self, name: Optional[str] = None) -> None:
        r"""
        Edge monad.
        """
        self.name = str(hex(id(self))) if name is None else str(name)
        return

    def __repr__(self) -> str:
        return f'<{self.name}>'


class Frame(object):
    r"""Multi-graph: ({Node}, {Edge}, R_n, R_e), where R_n: Node -> [Edge] while
    R_e: Edge -> [Node]

    All methods with side-effects should return `None`.
    """

    def __init__(self):
        self._node2edges = OrderedDict()  # type: OrderedDict[Node, list[Edge]]
        self._edge2nodes = OrderedDict()  # type: OrderedDict[Edge, list[Node]]
        return

    @property
    def nodes(self) -> list[Node]:
        return list(self._node2edges.keys())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edge2nodes.keys())

    def order(self, obj: Node | Edge) -> int:
        if isinstance(obj, Node):
            dct = self._node2edges
        elif isinstance(obj, Edge):
            dct = self._edge2nodes
        else:
            raise TypeError(
                f'Only define order for Node or Edge, not {type(obj)}.')

        assert obj in dct
        return len(dct[obj])

    @property
    def open_edges(self):
        return [e for e in self.edges if self.order(e) == 1]

    @property
    def end_points(self):
        return [n for n in self.nodes if self.order(n) == 1]

    def add_nodes(self,
                  nodes: list[Node],
                  edge: Optional[Edge] = None) -> None:
        assert nodes
        if edge is None:
            edge = Edge(name='-'.join(n.name for n in nodes))

        if edge not in self._edge2nodes:
            self._edge2nodes[edge] = []
        self._edge2nodes[edge] += nodes
        for n in nodes:
            if n not in self._node2edges:
                self._node2edges[n] = []
            self._node2edges[n] += [edge]

        return

    def add_edges(self,
                  edges: list[Edge],
                  node: Optional[Node] = None) -> None:
        assert edges
        if node is None:
            node = Node(name='*'.join(n.name for n in edges))

        if node not in self._node2edges:
            self._node2edges[node] = []
        self._node2edges[node] += edges
        for e in edges:
            if e not in self._edge2nodes:
                self._edge2nodes[e] = []
            self._edge2nodes[e] += [node]

        return

    def near_node(self, node: Node) -> list[Edge]:
        return self._node2edges[node]

    def near_edge(self, edge: Edge) -> list[Node]:
        return self._edge2nodes[edge]

    def next_nodes(self, node: Node) -> list[Node]:
        assert node in self.nodes

        ans = list()
        for e in self._node2edges[node]:
            ans += [n for n in self._edge2nodes[e] if n is not node]

        return ans

    def next_edges(self, edge: Edge) -> list[Edge]:
        assert edge in self.edges

        ans = list()
        for n in self._edge2nodes[edge]:
            ans += [e for e in self._node2edges[n] if e is not edge]

        return ans

    @property
    def node_graph(self) -> dict[Node, list[Node]]:
        """
        Undirectional node graph in dictionary.
        """
        return {n: self.next_nodes(n) for n in self.nodes}

    def find_link(self, n1: Node, n2: Node):
        es1 = self._node2edges[n1]
        es2 = self._node2edges[n2]
        cap = set(es1) & set(es2)

        if len(cap) == 1:
            e = cap.pop()
            i = self._node2edges[n1].index(e)
            j = self._node2edges[n2].index(e)
            return (n1, i, n2, j)

        else:
            raise NotImplementedError(
                f'Not support between {n1} and {n2} non-direct case and multi-graph.'
            )

    def node_visitor(self, start: Node, method: Literal['DFS', 'BFS'] = 'DFS'):
        return list(iter_visitor(start, self.next_nodes, method=method))

    def link_visitor(self, start: Node):
        nodes = [n for n in iter_round_visitor(start, self.next_nodes)]
        return [self.find_link(n1, n2) for n1, n2 in pairwise(nodes)]

    def visitor(self, start: Node):

        def _r(obj: Node | Edge) -> list[Edge | Node]:
            if isinstance(obj, Node):
                dct = self._node2edges
            else:
                assert isinstance(obj, Edge)
                dct = self._edge2nodes
            return dct[obj]

        return list(iter_round_visitor(start, _r))


class Network(object):
    """Network is a Frame with valuation for each node.
    """
    def __init__(self, frame: Frame) -> None:
        """
        Args:
            frame: Topology of the tensor network;

        """

        self.frame = frame
        self.dofs = frame.end_points
        self._dims = dict()  # type: dict[Edge, int]
        self._valuation = dict()  # type: dict[Node, Array]
        return

    def claim_array(self, node: Node, array: Array) -> None:
        assert node in self.frame.nodes and node not in self.frame.end_points
        assert self.frame.order(node) == array.ndim

        for i, e in enumerate(self.frame.near_node(node)):
            if e in self._dims:
                assert self._dims[e] == array.shape[i]
            else:
                self._dims[e] = array.shape[i]
        return

    def shape(self, node: Node) -> list[Optional[int]]:
        assert node in self.frame.nodes

        node_shape = [None] * self.frame.order(node)

        for i, e in enumerate(self.frame.near_node(node)):
            if e in self._dims:
                node_shape[i] = self._dims[e]
        return node_shape

    def __getitem__(self, node: Node) -> Array:
        return self._valuation[node]

    def __setitem__(self, node: Node, array: Array) -> None:
        self.claim_array(node, array)
        self._valuation[node] = array
        return

    def __delitem__(self, node: Node):
        del self._valuation[node]
        return

    def fill_zeros(self, dims: Optional[dict[Edge, int]] = None) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the dimension for each Edge in dims (default is 1).
        """
        if dims is None:
            dims = dict()

        for n in self.frame.nodes:
            if n not in self.dofs and n not in self._valuation:
                shape = [
                    self._dims.get(e, dims.get(e, 1))
                    for e in self.frame.near_node(n)
                ]
                self[n] = zeros(shape)

        return


class State(Network):
    """
    State is a Network with Tree-shape and root.
    """

    def __init__(self, frame: Frame, root: Node) -> None:
        super().__init__(frame)

        self._root = root
        self._depthes = None    # type: Optional[dict[Node, int]]
        self._axes = None  # type: Optional[dict[Node, Optional[int]]]
        return

    @property
    def root(self) -> Node:
        return self._root

    @root.setter
    def root(self, value: Node) -> None:
        assert value in self.frame.nodes
        self._root = value
        self._depthes = None
        self._axes = None
        return

    @property
    def depthes(self) -> dict[Node, int]:
        if self._depthes is None:
            ans = depth_dict(self.root, self.frame.next_nodes)
            self._depthes = ans
        else:
            ans = self._depthes
        return ans

    @property
    def axes(self) -> dict[Node, Optional[int]]:
        if self._axes is None:
            ans = {self.root: None}
            for m, i, n, j in self.frame.link_visitor(self.root):
                if m in ans and n not in ans:
                    ans[n] = j
            self._axes = ans
        else:
            ans = self._axes
        return ans
