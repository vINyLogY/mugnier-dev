# coding: utf-8
r"""Data structure for topology of tensors in a network

"""

from __future__ import annotations
from collections import OrderedDict



from typing import Iterable, Optional

from mugnier.libs.utils import T, Cached, iter_visitor, lazyproperty
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
            raise TypeError(f'Only define order for Node or Edge, not {type(obj)}.')

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

    def near_edge(self, edge: Edge) ->  list[Node]:
        return self._edge2nodes[edge]

    def next_nodes(self,
                   node: Node) -> list[Node]:
        assert node in self.nodes

        ans = list()
        for e in self._node2edges[node]:
            ans += [n for n in self._edge2nodes[e] if n is not node]

        return ans

    def next_edges(self,
                   edge: Edge) -> list[Edge]:
        assert edge in self.edges

        ans = list()
        for n in self._edge2nodes[edge]:
                ans += [e for e in self._node2edges[n] if e is not edge]

        return ans

    @property
    def node_graph(self) -> dict[Node, list[Node]]:
        return {n: self.next_nodes(n) for n in self.nodes}




class Network(object):

    def __init__(self, frame: Frame) -> None:
        """
        Args:
            frame: Topology of the tensor network;

        """

        self.frame = frame
        self.dofs = frame.end_points
        self._dims = dict  # type: dict[Edge, int]
        self._valuation = dict()  # type: dict[Node, Array]
        return

    def claim_array(self, node: Node, array: Array) -> None:
        assert node in self.frame
        assert self.frame.order(node) == array.ndim

        for i, e in enumerate(self.frame.near(node)):
            if e in self._dims:
                assert self._dims[e] == array.shape[i]
            else:
                self._dims[e] = array.shape[i]
        return

    def shape(self, node: Node) -> list[Optional[int]]:
        assert node in self.frame.nodes

        node_shape = [None] * self.frame.order(node)

        for i, e in enumerate(self.frame.near(node)):
            if e in self._dims:
                node_shape[i] = self._dims[e]
        return node_shape

    def __getitem__(self, node: Node) -> Array:
        return self._valuation[node]

    def __setitem__(self, node: Node, array: Array) -> None:
        self.claim_array(node, array)
        self._valuation[node] = asarray(array)
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
            if n not in self._valuation:
                shape = [
                    self._dims.get(e, dims.get(e, 1))
                    for e in self.frame.near(n)
                ]
                self[n] = zeros(shape)

        return


# class State(Network):
#     """
#     State with Tree restricted shape and valuation
#     """

#     def __init__(self, frame: Frame, root: Node) -> None:
#         super().__init__(frame)

#         self.root = root
#         return

#     @lazyproperty
#     def depth(self) -> dict[Node, int]:
#         d = 0
#         dct = dict()
#         visited = set()
#         stack = set([self.root])
#         neighbor = partial(self.frame.next_nodes)
#         while len(stack) > 0:
#             dct.update({n: d for n in stack})
#             visited |= stack
#             stack = set().union(*(set(neighbor(s)) for s in stack)) - visited
#             d += 1

#         return dct

#     @lazyproperty
#     def axes(self) -> dict[Node, OptEdge]:
#         root = self.root
#         near = self.frame.near

#         dct = {root: None}
#         visited = set([root])
#         stack = set([root])

#         while len(stack) > 0:
#             n = stack.pop()

#             new_items = {
#                 child: e
#                 for e in near(n) for child in near(e) if child not in visited
#             }

#             dct.update(new_items)
#             new_nodes = set(new_items.keys())
#             visited |= new_nodes
#             stack |= new_nodes

#         return dct

#     def annotated_array(self,
#                         edge: Edge,
#                         conj: bool = False) -> Tuple[ArrayLike, int]:

#         node = max(self.frame.near(edge), self.depth.get)

#         i = self.frame.near(node).index(edge)
#         array = self._valuation[node]
#         if conj:
#             array = np.conj(array)
#         return array, i



