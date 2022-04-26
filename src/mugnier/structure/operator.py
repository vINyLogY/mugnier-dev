# coding: utf-8

from itertools import chain
from typing import Iterable

from mugnier.libs.backend import (MAX_EINSUM_AXES, Array, OptArray, eye, opt_einsum, opt_sum, optimize, np)
from mugnier.structure.network import End


class SumProdOp(object):
    stacked_ax = 0

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
        return array.reshape([1] + shape).expand([self.n_terms] + shape)
