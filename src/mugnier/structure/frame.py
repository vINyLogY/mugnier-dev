# coding: utf-8

from platform import node
from typing import Iterable, Literal, Optional
from mugnier.libs.utils import huffman_tree

from mugnier.structure.network import Frame, Node, End, Point


class Singleton(Frame):

    def __init__(self, ends: list[End]) -> None:
        super().__init__()
        r = Node(f'0')
        for e in ends:
            self.add_link(r, e)
        return


class TensorTrain(Frame):
    """
    Attr:
        train: list of the TensorTrain nodes
    """

    def __init__(self, ends: list[End]) -> None:
        super().__init__()
        dof = len(ends)
        assert dof > 0
        nodes = [Node(f'{i}') for i in range(dof)]
        for n, e in enumerate(ends):
            self.add_link(nodes[n], e)
        if dof > 1:
            for n in range(dof - 1):
                self.add_link(nodes[n], nodes[n + 1])
        self.train = nodes
        return


class TensorTrainMCTDH(Frame):
    """
    Attr:
        nodes: list of the TensorTrain nodes
    """

    def __init__(self, ends: list[End]) -> None:
        super().__init__()
        dof = len(ends)
        assert dof > 0
        nodes = [Node(f'T{i}') for i in range(dof)]
        spfs = [Node(f'S{i}') for i in range(dof)]
        for n, e in enumerate(ends):
            self.add_link(spfs[n], e)
            self.add_link(nodes[n], spfs[n])
        if dof > 1:
            for n in range(dof - 1):
                self.add_link(nodes[n], nodes[n + 1])
        return


class Tree(Frame):
    """
    Attrs:
        root : a default root Node of the tree.
    """
    def __init__(self, graph: dict[Node, list[Point]], root: Node):
        super().__init__()
        self.root = root
        for n, children in graph.items():
            for child in children:
                self.add_link(n, child)
        return


class MultiLayerMCTDH(Tree):

    def __init__(self, ends: list[End],
                 importances: Optional[list[int]] = None,
                 n_ary: int = 2) -> None:
        """
        Generate a Tree-like Frame (ML-MCTDH) from a node graph.
        """
        class new_node(Node):
            _counter = 0

            def __init__(self) -> None:
                cls = type(self)
                super().__init__(f'T{cls._counter}')
                cls._counter += 1

        graph, root = huffman_tree(ends, new_node, importances=importances, n_ary=n_ary)
        super().__init__(graph, root)
        return


class TwoLayerMCTDH(Tree):
    """
    Simple two layer tree (in MCTDH) without mode combinations.
    """

    def __init__(self, ends: list[End]) -> None:
        root = Node('0')
        spfs_dict = {Node(e.name): e for e in ends}
        graph = {root: list(spfs_dict.keys())}
        graph.update(spfs_dict)
        super().__init__(graph, root)
        return
