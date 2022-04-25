# coding: utf-8

from itertools import chain
from typing import Optional, Tuple
from mugnier.libs.backend import OptArray, opt_einsum, opt_exp, MAX_EINSUM_AXES
from mugnier.structure.network import End, Node, Point, State
from mugnier.structure.operator import SumProdOp


def transform(tensor: OptArray, op_dict: dict[int, Optional[OptArray]]) -> OptArray:
    op_dict = {j: a for j, a in op_dict.items() if a is not None}
    order = tensor.ndim
    n = len(op_dict)
    ax_list, mat_list = zip(*op_dict.items())
    assert order + n < MAX_EINSUM_AXES

    args = [(tensor, list(range(order)))]
    args.extend((mat_list[i], [order + i, ax_list[i]]) for i in range(n))
    ans_axes = [order + ax_list.index(j) if j in ax_list else j for j in range(order)]
    args.append((ans_axes, ))

    ans = opt_einsum(*chain(*args))

    return ans


class SumProdEOM(object):
    r"""Solve the equation wrt Sum-of-Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self, op: SumProdOp, state: State) -> None:
        assert op.ends == state.ends
        self._op = op
        self._state = state
        self._mean_fields = dict()  # type: dict[Tuple[Node, int], Optional[OptArray]]
        self._reduced_densities = dict()  # type: dict[Tuple[Node, int], OptArray]
        return

    def single_diff(self, node: Node) -> OptArray:
        array = self._state[node]
        op_list = {i: self.mean_field(node, i) for i in range(array.ndim)}

        if node is not self._state.root:
            raise NotImplementedError

        array = self._op.transform(array, op_list)
        return self._op.reduce(array)

    def single_node_split_propagate(self, node: Node, dt: float) -> None:
        array = self._state[node]
        ops = {i: self.mean_field(node, i) for i in range(array.ndim)}

        if node is not self._state.root:
            raise NotImplementedError

        us = {i: opt_exp(-0.5j * dt * a) for i, a in ops.items() if a is not None}

        n_terms = len(self._op.ends)

        for n in range(n_terms):
            un = {i: a[n] for i, a in us.items()}
            array = transform(array, un)

        for n in reversed(range(n_terms)):
            un = {i: a[n] for i, a in us.items()}
            array = transform(array, un)

        self._state[node] = array
        return

    def mean_field(self, p: Point, i: int) -> Optional[OptArray]:
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
        a = self._state[p]
        op_list = {j: self.mean_field(p, j) for j in range(a.ndim) if j != i}
        return self._op.overlap(self._op.transform(a, op_list), self._op.extend(a.conj()), ax=i)

    def reduced_densities(self, p: Node) -> OptArray:
        raise NotImplementedError

    def propagator(steps=None, internal=0.1, start=0.0):
        pass
