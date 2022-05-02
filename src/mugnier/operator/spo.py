# coding: utf-8

from itertools import chain, count
from typing import Callable, Generator, Iterable, Optional, Tuple

import numpy as np
from mugnier.libs.backend import (MAX_EINSUM_AXES, Array, OptArray, eye, opt_einsum, opt_sum, optimize)
from mugnier.state.frame import End, Node, Point
from mugnier.state.model import CannonialModel


class SumProdOp:

    def __init__(self, op_list: list[dict[End, Array]]) -> None:
        n_terms = len(op_list)
        tensors = dict()
        dims = dict()

        for term in op_list:
            for e, a in term.items():
                assert a.ndim == 2 and a.shape[0] == a.shape[1]
                if e in dims:
                    assert dims[e] == a.shape[0]
                else:
                    dims[e] = a.shape[0]

        # Densify the op_list along DOFs to axis-0
        tensors = {e: [eye(dim, dim)] * n_terms for e, dim in dims.items()}
        for n, term in enumerate(op_list):
            for e, a in term.items():
                tensors[e][n] = a

        self.n_terms = n_terms
        self._dims = dims  # type: dict[End, int]
        self._valuation = {e: optimize(np.stack(a, axis=0)) for e, a in tensors.items()}  # type: dict[End, OptArray]
        return

    def __getitem__(self, key: End) -> OptArray:
        return self._valuation[key]

    @property
    def ends(self) -> set[End]:
        return set(self._dims.keys())

    @staticmethod
    def transform(tensors: OptArray, op_dict: dict[int, OptArray]) -> Iterable:
        order = tensors.ndim - 1

        n = len(op_dict)
        ax_list = list(sorted(range(order), key=(lambda ax: tensors.shape[ax + 1])))
        mat_list = [op_dict[ax] for ax in ax_list]
        op_ax = 1 + order + n

        assert op_ax < MAX_EINSUM_AXES

        args = [(tensors, [op_ax] + list(range(order)))]
        args.extend((mat_list[i], [op_ax, order + i, ax_list[i]]) for i in range(n))
        ans_axes = [op_ax] + [order + ax_list.index(ax) if ax in ax_list else ax for ax in range(order)]
        args.append((ans_axes,))

        ans = opt_einsum(*chain(*args))

        return ans

    @staticmethod
    def reduce(array: OptArray) -> OptArray:
        return opt_sum(array, 0)

    def expand(self, array: OptArray) -> OptArray:
        shape = list(array.shape)
        return array.unsqueeze(0).expand([self.n_terms] + shape)


class Integrator(object):
    r"""Solve the equation wrt Sum-of-Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self, op: SumProdOp, state: CannonialModel) -> None:
        assert op.ends == state.ends
        self._op = op
        self._state = state

        self._mean_fields = dict()  # type: dict[Tuple[Node, int], OptArray]
        self._reduced_densities = dict()  # type: dict[Node, OptArray]
        return

    def split_diff(self, node: Node) -> Callable[[OptArray], OptArray]:
        if node is not self._state.root:
            raise NotImplementedError

        order = self._state.frame.order(node)
        op_list = {i: self.mean_field(node, i) for i in range(order)}
        expand = self._op.expand
        transform = self._op.transform
        reduce = self._op.reduce

        def _diff(array: OptArray) -> OptArray:
            assert array.ndim == order
            return reduce(transform(expand(array), op_list))

        return _diff

    def split_diff_op(self, node: Node) -> OptArray:
        if node is not self._state.root:
            raise NotImplementedError

        array = self._state[node]
        dims = array.shape
        order = array.ndim

        ax_list = list(sorted(range(order), key=(lambda ax: dims[ax])))
        mat_list = [self.mean_field(node, ax) for ax in ax_list]

        op_ax = 2 * order

        from_axes = list(range(order))
        to_axes = list(range(order, 2 * order))

        args = [(mat_list[i], [op_ax, order + ax_list[i], ax_list[i]]) for i in range(order)]
        args.append((to_axes + from_axes,))
        diff = opt_einsum(*chain(*args))

        return diff

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
        a10 = 1.0 / 3.0
        a20, a21 = -1.0 / 3.0, 1.0
        a30, a31, a32 = 1.0, -1.0, 1.0

        k0 = diff(y0)
        k1 = diff(y0 + dt * (a10 * k0))
        k2 = diff(y0 + dt * (a20 * k0 + a21 * k1))
        k3 = diff(y0 + dt * (a30 * k0 + a31 * k1 + a32 * k2))

        c0, c1 = 1.0 / 8.0, 3.0 / 8.0

        return (y0 + c0 * (k0 + k3) + c1 * (k1 + k2))

    def propagator(self, steps: Optional[int] = None, interval: float = 1.0) -> Generator[float, None, None]:
        root = self._state.root

        for n in count():
            if steps is not None and n >= steps:
                break
            time = n * interval
            yield (time, self._state)

            func = self.split_diff(root)
            new_array = self.rk4(func, self._state[root], interval)

            self._state.opt_update(root, new_array)
