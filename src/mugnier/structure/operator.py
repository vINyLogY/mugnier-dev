# coding: utf-8

from itertools import chain

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
        self._dims = dims
        self._valuation = {e: optimize(np.stack(a, axis=0)) for e, a in tensors.items()}  # type: dict[End, OptArray]
        return

    def __getitem__(self, key: End) -> OptArray:
        return self._valuation[key]

    @property
    def ends(self) -> set[End]:
        return set(self._dims.keys())

    @staticmethod
    def transform(tensors: OptArray, op_dict: dict[int, OptArray]) -> OptArray:
        order = tensors.ndim - 1
        n = len(op_dict)
        ax_list, mat_list = zip(*op_dict.items())
        op_ax = 1 + order + n
        assert op_ax < MAX_EINSUM_AXES

        args = [(tensors, [op_ax] + list(range(order)))]
        args.extend((mat_list[i], [op_ax, order + i, ax_list[i]]) for i in range(n))
        ans_axes = [op_ax] + [order + ax_list.index(j) if j in ax_list else j for j in range(order)]
        args.append((ans_axes, ))

        ans = opt_einsum(*chain(*args))

        return ans

    @staticmethod
    def trace(tensors1: OptArray, tensors2: OptArray, axis_left: int) -> OptArray:
        assert tensors1.shape == tensors2.shape
        order = tensors1.ndim - 1
        op_ax = order
        assert op_ax + 2 < MAX_EINSUM_AXES

        axes1 = [op_ax] + list(range(order))
        axes2 = [op_ax] + list(range(order))
        axes1[axis_left + 1] = op_ax + 1
        axes2[axis_left + 1] = op_ax + 2

        ans = opt_einsum(tensors1, axes1, tensors2, axes2, [op_ax, op_ax + 1, op_ax + 2])
        return ans

    @staticmethod
    def overlap(tensors1: OptArray, tensors2: OptArray) -> OptArray:
        assert tensors1.shape == tensors2.shape
        order = tensors1.ndim - 1
        op_ax = order
        assert op_ax < MAX_EINSUM_AXES

        axes1 = [op_ax] + list(range(order))
        axes2 = [op_ax] + list(range(order))

        ans = opt_einsum(tensors1, axes1, tensors2, axes2, [op_ax])
        return ans

    @staticmethod
    def reduce(array: OptArray) -> OptArray:
        return opt_sum(array, 0)

    def expand(self, array: OptArray) -> OptArray:
        return array.reshape([1] + array.shape).expand([self.n_terms] + array.shape)
