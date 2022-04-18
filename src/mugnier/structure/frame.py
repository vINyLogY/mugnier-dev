# coding: utf-8

from typing import Iterable, Literal, Optional
from mugnier.libs.utils import huffman_tree

from mugnier.structure.network import Frame, Node, Edge


class Singleton(Frame):

    def __init__(self, edges: list[Edge]) -> None:
        super().__init__()
        self.edges(list(edges), node=Node(name='0'))

        return


class TensorTrain(Frame):
    """
    Attr:
        nodes: list of the TensorTrain nodes
    """

    def __init__(self, edges: list[Edge]) -> None:
        super().__init__()
        dof = len(edges)
        assert dof > 0
        nodes = [Node(f'{i}') for i in range(dof)]
        for n, edge in enumerate(edges):
            self.add_edges([edge], node=nodes[n])
        if dof > 1:
            for n in range(dof - 1):
                self.add_nodes([nodes[n], nodes[n + 1]], edge=Edge(f'{n}-{n+1}'))
        return


class TensorTrainMCTDH(Frame):
    """
    Attr:
        nodes: list of the TensorTrain nodes
    """

    def __init__(self, edges: list[Edge]) -> None:
        super().__init__()
        dof = len(edges)
        assert dof > 0
        nodes = [Node(f'T{i}') for i in range(dof)]
        spfs = [Node(f'S{i}') for i in range(dof)]
        for n, edge in enumerate(edges):
            self.add_edges([edge], node=spfs[n])
            self.add_nodes([nodes[n], spfs[n]])
        if dof > 1:
            for n in range(dof - 1):
                self.add_nodes([nodes[n], nodes[n + 1]], edge=Edge(f'{n}-{n+1}'))
        return


class Tree(Frame):
    def __init__(self, graph: dict[Node, list[Node | Edge]], root: Node):
        super().__init__()
        self.root = root
        for n, children in graph.items():
            for child in children:
                if isinstance(child, Edge):
                    self.add_nodes([n], edge=child)
                else:
                    self.add_nodes([n, child])
        return


class MultiLayerMCTDH(Tree):

    def __init__(self, edges: list[Edge],
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

        graph, root = huffman_tree(edges, new_node, importances=importances, n_ary=n_ary)
        super().__init__(graph, root)
        return


class TwoLayerMCTDH(Tree):
    """
    Simple two layer tree (in MCTDH) without mode combinations.
    """

    def __init__(self, edges: list[Edge]) -> None:
        root = Node('0')
        spfs_dict = {Node(e.name): e for e in edges}
        graph = {root: list(spfs_dict.keys())}
        graph.update(spfs_dict)
        super().__init__(graph, root)
        return
