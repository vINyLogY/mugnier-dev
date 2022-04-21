# coding: utf-8
from platform import node
from typing import Tuple
from mugnier.libs.utils import depth_dict, huffman_tree
from mugnier.structure.frame import MultiLayerMCTDH, TensorTrain
from mugnier.structure.network import Node, End, Point, State


def test_frame():
    N = 6
    dofs = [End(name='elec')] + [End(name=f'nuc_{i}') for i in range(N)]
    f1 = TensorTrain(dofs)

    print('Printing a TensorTrain frame...')
    print(f1._neighbor)

    def dim(p: Point, q: Point):
        if p is dofs[0] or q is dofs[0]:
            return 2
        elif p in dofs or q in dofs:
            return 20
        else:
            return 5

    s1 = State(f1, f1.train[0])
    s1.fill_eyes({(p, i): dim(p, q) for p, i, q, _ in f1.links})

    for n in f1.node_visitor(f1.train[0]):
        print(n, s1[n].shape)

    s1.to_gpu()

    for n in f1.node_visitor(f1.train[0]):
        print(n, s1[n].shape)


if __name__ == '__main__':
    test_frame()
