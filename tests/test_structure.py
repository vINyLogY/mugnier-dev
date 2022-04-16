# coding: utf-8
from platform import node
from mugnier.structure.network import Node, Edge
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


    print('Printing a manual added tensor train frame with open edges')
    print(f1.node_graph)

    for e in f1.open_edges:
        f1.add_nodes([Node(name=e.name)], edge=e)

    print('Printing a manual added tensor train frame afted fix with nodes')
    print(f1.node_graph)


if __name__ == '__main__':
    test_elements()

