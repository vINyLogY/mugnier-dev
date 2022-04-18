# coding: utf-8
from platform import node
from mugnier.libs.utils import depth_dict, huffman_tree
from mugnier.structure.frame import MultiLayerMCTDH, TensorTrain, Tree
from mugnier.structure.network import Node, Edge, State
from mugnier.structure.network import Frame


def test_elements():
    N = 10
    f1 = Frame()

    for n in range(N):
        f1.add_edges([Edge(name=f'b{n}')], node=Node(name=f'{n}'))

    f1.add_edges([Edge(name='0-1')], node=Node(name='0'))
    for n in range(1, N - 1):
        f1.add_edges([Edge(name=f'{n-1}-{n}'),
                      Edge(name=f'{n}-{n+1}')],
                     node=Node(name=f'{n}'))
    f1.add_edges([Edge(name=f'{N-2}-{N-1}')], node=Node(name=f'{N-1}'))

    print('Printing a manual added tensor train frame...')
    print(f1.node_graph)

    f2 = Frame()

    for i in range(N):
        f2.add_edges(
            [Edge(name=f'b{i}'), Edge(name=f'b{i}')], node=Node(name=f'{i}'))

    f2.add_edges([Edge(name='0-1')], node=Node(name='0'))
    for n in range(1, N - 1):
        f2.add_edges([Edge(name=f'{n-1}-{n}'),
                      Edge(name=f'{n}-{n+1}')],
                     node=Node(name=f'{n}'))
    f2.add_edges([Edge(name=f'{N-2}-{N-1}')], node=Node(name=f'{N-1}'))
    print('Printing a manual added tensor train operator eframe...')
    print(f2.edges)


def test_frame():
    N = 10
    dofs = [Edge(name='elec')] + [Edge(name=f'nuc_{i}') for i in range(N)]
    f1 = TensorTrain(dofs)

    print('Printing a TensorTrain frame...')
    print(f1.node_graph)

    N = 8
    dofs = [Edge(name=f'nuc_{i}') for i in range(N)]

    f1 = MultiLayerMCTDH(dofs, n_ary=2)

    print('Printing a HuffmanTree frame...')
    print(f1.node_graph)
    print(f1.visitor(f1.root))
    print(depth_dict(f1.root, f1.next_nodes))

    def dims(dof):
        if dof is Node(name='elec'):
            return 2
        elif dof in dofs:
            return 20
        else:
            return 5

    s1 = State(f1, f1.root)
    s1.fill_zeros({e: dims(e) for e in f1.edges})
    print(f1.node_graph)
    for n in f1.nodes:
        print(n, s1[n].shape)

    print(f'Root @ {s1.root}')
    print(s1.axes)
    print(s1.depthes)


if __name__ == '__main__':
    test_frame()
