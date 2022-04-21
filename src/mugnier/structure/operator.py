# coding: utf-8

from itertools import chain
from typing import Optional, Tuple


from mugnier.libs.backend import MAX_EINSUM_AXES, Array, to_gpu, einsum, eye, stack
from mugnier.structure.network import End, Node, State


class SumProdOp(object):
    stacked_ax = 0

    def __init__(self, op_list: list[dict[End, Array]]) -> None:
        n_max = len(op_list)
        tensors = dict()

        # Densify the op_list
        for term in op_list:
            for e, a in term.items():
                assert a.ndim == 2 and a.shape[0] == a.shape[1]
                tensors[e] = [eye(a.shape[0], a.shape[1])] * n_max

        for n, term in enumerate(op_list):
            for e, a in term.items():
                tensors[e][n] = a

        self._valuation = {e: stack(tensors[e])}
        return

    def __getitem__(self, key: End) -> Optional[Array]:
        return self._valuation.get(key)

    @staticmethod
    def transform(tensor: Array, op_list: dict[int, Array]):
        order = tensor.ndim
        n = len(op_list)
        ax_list, mat_list = zip(*sorted(mat_list.items()))
        op_ax = 1 + order + n
        assert op_ax < MAX_EINSUM_AXES

        args = [(tensor, list(range(order)))]
        args.extend((mat_list[i], [op_ax, order + i, ax_list[i]]) for i in range(n))
        ans_axes = [op_ax] + [order + ax_list.index(j) if j in ax_list else j for j in range(order)]
        args.append((ans_axes, ))

        ans = einsum(*chain(*args)).detach()

        return ans

    @staticmethod
    def overlap(tensor1: Array, tensor2: Array, ax: Optional[int] = None):
        assert tensor1.shape == tensor2.shape
        order = len(tensor1.shape)
        op_ax = order
        assert order + 2 < MAX_EINSUM_AXES

        axes1 = list(range(order))
        axes1[0] = order
        axes2 = list(range(order))
        axes2[0] = order

        if ax is not None:
            axes1[ax + 1] = order + 1
            axes2[ax + 1] = order + 2

        ans = einsum(tensor1, axes1, tensor2, axes2)
        return ans


class SumProdEOM(object):
    r"""Solve the equation wrt Sum-of-Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self, op: SumProdOp, state: State) -> None:
        self._op = op
        self._state = state
        self._mean_fields = dict()  # type: dict[Tuple[Node, int], Optional[Array]]
        self._reduced_densities = dict()  # type: dict[Tuple[Node, int], Array]
        return

    def single_diff(self, node: Node) -> Array:
        state = self._state
        op = [(i, self.mean_field(state)[node, i]) for i, _ in range(state.frame.order(node))]

        if node is not state.root:
            raise NotImplementedError

    def single_propagator(self, node: Node) -> Array:
        pass

    def mean_fields(self, node: Node, i: int) -> Optional[Array]:
        edge = self._state.frame.near_node(node)[i]
        if edge in self._state.dofs:
            return self._op[edge]

        try:
            return self._mean_fields[node, i]
        except IndexError:
            _n = self._state.frame.next_nodes(node)[i]
            _a = self._state[_n]
            tmp = self._op.transform(_a, {_i: self.mean_fields(_n, _i) for _i in range(_a.ndim)})

            ans = self._op.overlap(tmp)

            self._mean_fields[(node, i)] = ans
            return ans

    def reduced_densities(self, node: Node) -> Array:
        raise NotImplementedError
