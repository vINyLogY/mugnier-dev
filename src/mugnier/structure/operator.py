# coding: utf-8

from itertools import chain
from typing import Optional, Tuple

from torch import tensor


from mugnier.libs.backend import MAX_EINSUM_AXES, Array, OptArray, opt_sum, optimize, opt_einsum, eye, stack
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
        self._dims = dims
        self._valuation = {e: optimize(stack(a, axis=0)) for e, a in tensors.items()}  # type: dict[End, OptArray]
        return

    def __getitem__(self, key: End) -> Optional[OptArray]:
        return self._valuation.get(key)

    @property
    def ends(self) -> set[End]:
        return set(self._dims.keys())

    @staticmethod
    def transform(tensor: OptArray, op_dict: dict[int, Optional[OptArray]]) -> OptArray:
        op_dict = {j: a for j, a in op_dict.items() if a is not None}
        order = tensor.ndim
        n = len(op_dict)
        ax_list, mat_list = zip(*op_dict.items())
        op_ax = 1 + order + n
        assert op_ax < MAX_EINSUM_AXES

        args = [(tensor, list(range(order)))]
        args.extend((mat_list[i], [op_ax, order + i, ax_list[i]]) for i in range(n))
        ans_axes = [op_ax] + [order + ax_list.index(j) if j in ax_list else j for j in range(order)]
        args.append((ans_axes, ))

        ans = opt_einsum(*chain(*args))

        return ans

    @staticmethod
    def overlap(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
        assert tensor1.shape == tensor2.shape
        order = tensor1.ndim
        assert order + 2 < MAX_EINSUM_AXES

        axes1 = list(range(order))
        axes1[0] = order
        axes2 = list(range(order))
        axes2[0] = order
        axes1[ax + 1] = order + 1
        axes2[ax + 1] = order + 2

        ans = opt_einsum(tensor1, axes1, tensor2, axes2, [order, order + 1, order + 2])
        return ans

    @staticmethod
    def reduce(array: OptArray) -> OptArray:
        return opt_sum(array, 0)

    def expand(self, array: OptArray) -> OptArray:
        return array.reshape([1] + array.shape).expand([self.n_terms] + array.shape)
