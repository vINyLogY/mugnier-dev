# coding: utf-8
from typing import Tuple
from mugnier.libs.backend import array
from mugnier.structure.frame import MultiLayerMCTDH, Singleton, TensorTrain
from mugnier.structure.network import Node, End, Point, State
from mugnier.structure.operator import SumProdOp





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


def test_operator():
    import numpy as np
    N = 4
    dofs = [End(name='elec')] + [End(name=f'nuc_{i}') for i in range(N)]
    f = Singleton(dofs)

    dim = 20

    ops = []
    he = array([[-0.5, 0.1], [0.1, 0.5]])
    ops.append({dofs[0]: he})
    sigma_z = array([[-0.5, 0.], [0., 0.5]])
    sqrt_n = np.diag(np.sqrt(np.arange(dim)))
    x = sqrt_n @ np.eye(dim, k=-1) + np.eye(dim, k=1) @ sqrt_n
    for e in dofs[1:]:
        ops.append({dofs[0]: sigma_z, e: x})

    op = SumProdOp(ops)
    for e, a in op._valuation.items():
        print((e, a.shape))




if __name__ == '__main__':
    test_frame()
    test_operator()
