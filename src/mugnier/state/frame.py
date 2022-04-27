# coding: utf-8
r"""Data structure for topology of tensors in a network

"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Optional, Tuple
from weakref import WeakValueDictionary

from mugnier.libs.utils import iter_round_visitor, iter_visitor


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


class Frame:

    def __init__(self):
        self._neighbor = dict()  # type: dict[Point, list[Point]]
        self._duality = dict()  # type: dict[Tuple[Point, int], Tuple[Point, int]]
        return

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
