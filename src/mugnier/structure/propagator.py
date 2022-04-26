# coding: utf-8

from functools import partial
from itertools import chain, count
from math import prod
from typing import Callable, Generator, Optional, Tuple
from mugnier.libs.backend import OptArray, opt_einsum, MAX_EINSUM_AXES, np
from mugnier.structure.network import End, Node, Point, State
from mugnier.structure.operator import SumProdOp

import torch


def transform(tensor: OptArray, op_dict: dict[int, OptArray]) -> OptArray:
    order = tensor.ndim
    assert len(op_dict) == order
    assert 2 * order < MAX_EINSUM_AXES

    ax_list = list(sorted(range(order), key=(lambda ax: tensor.shape[ax])))
    mat_list = [op_dict[ax] for ax in ax_list]

    args = [(tensor, list(range(order)))]
    args.extend((mat_list[i], [order + ax_list[i], ax_list[i]]) for i in range(order))
    args.append(([order + ax_list[i] for i in range(order)],))

    return opt_einsum(*chain(*args))


class SumProdEOM(object):
    r"""Solve the equation wrt Sum-of-Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self, op: SumProdOp, state: State) -> None:
        assert op.ends == state.ends
        self._op = op
        self._state = state

        self._mean_fields = dict()  # type: dict[Tuple[Node, int], OptArray]
        self._reduced_densities = dict()  # type: dict[Node, OptArray]
        return

    def single_diff(self, node: Node) -> Callable[[OptArray], OptArray]:
        order = self._state.frame.order(node)
        op_list = {i: self.mean_field(node, i) for i in range(order)}
        expand = self._op.expand
        transform = self._op.transform
        reduce = self._op.reduce

        def _diff(array: OptArray) -> OptArray:
            assert array.ndim == order
            return reduce(transform(expand(array), op_list))

        return _diff

    # def single_node_split_propagate(self, node: Node, dt: float) -> None:
    #     array = self._state[node]
    #     ops = {i: self.mean_field(node, i) for i in range(array.ndim)}

    #     if node is not self._state.root:
    #         raise NotImplementedError

    #     us = {i: opt_exp(0.5 * dt * a) for i, a in ops.items()}

    #     n_terms = len(self._op.ends)

    #     for n in range(n_terms):
    #         un = {i: a[n] for i, a in us.items()}
    #         array = transform(array, un)

    #     for n in reversed(range(n_terms)):
    #         un = {i: a[n] for i, a in us.items()}
    #         array = transform(array, un)

    #     self._state.opt_update(node, array)
    #     return

    # def single_node_propagator(self, node: Node) -> OptArray:
    #     array = self._state[node]
    #     dims = array.shape
    #     order = array.ndim

    #     if node is not self._state.root:
    #         raise NotImplementedError

    #     ax_list = list(sorted(range(order), key=(lambda ax: dims[ax])))
    #     mat_list = [self.mean_field(node, ax) for ax in ax_list]

    #     op_ax = 2 * order

    #     from_axes = list(range(order))
    #     to_axes = list(range(order, 2 * order))

    #     args = [(mat_list[i], [op_ax, order + ax_list[i], ax_list[i]]) for i in range(order)]
    #     args.append((to_axes + from_axes,))
    #     diff = opt_einsum(*chain(*args))

    #     return diff

    def mean_field(self, p: Point, i: int) -> OptArray:
        q, j = self._state.frame.dual(p, i)

        if isinstance(q, End):
            return self._op[q]

        try:
            ans = self._mean_fields[(p, i)]
        except IndexError:
            ans = self._mean_field_with_node(q, j)
            self._mean_fields[(p, i)] = ans
        return ans

    def _mean_field_with_node(self, p: Node, i: int):
        a = self._op.expand(self._state[p])
        op_list = {j: self.mean_field(p, j) for j in range(a.ndim) if j != i}
        return self._op.trace(self._op.transform(a, op_list), self._op.extend(a.conj()), ax=i)

    def reduced_densities(self, p: Node) -> OptArray:
        try:
            ans = self._reduced_densities[p]
        except IndexError:
            raise NotImplementedError

        return ans

    @staticmethod
    def rk4(diff, y0, dt):
        k1 = diff(y0)
        k2 = diff(y0 + 0.5 * dt * k1)
        k3 = diff(y0 + 0.5 * dt * k2)
        k4 = diff(y0 + dt * k3)
        c1 = 1.0 / 6.0
        c2 = 1.0 / 3.0
        return (y0 + c1 * (k1 + k4) + c2 * (k2 + k3))

    def propagator(self,
                   steps: Optional[int] = None,
                   interval: float = 0.1) -> Generator[Tuple[float, State], None, None]:
        root = self._state.root

        for n in count():
            if steps is not None and n >= steps:
                break
            time = n * interval
            yield (time, self._state)

            func = self.single_diff(root)
            new_array = self.rk4(func, self._state[root], interval)

            self._state.opt_update(root, new_array)
