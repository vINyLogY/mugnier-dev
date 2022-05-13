# coding: utf-8

from functools import cache
from itertools import chain, count
from os import stat
from turtle import shape
from typing import Callable, Generator, Iterable, Optional, Tuple

import numpy as np
from torch import float_power
from mugnier.libs import backend
from mugnier.libs.backend import (MAX_EINSUM_AXES, Array, OptArray, eye, odeint, opt_einsum, opt_sum, opt_tensordot,
                                  optimize)
from mugnier.libs.utils import depths
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
        self.dims = dims  # type: dict[End, int]
        self._valuation = {e: optimize(np.stack(a, axis=0)) for e, a in tensors.items()}  # type: dict[End, OptArray]
        return

    def __getitem__(self, key: End) -> OptArray:
        return self._valuation[key]

    @property
    def ends(self) -> set[End]:
        return set(self.dims.keys())

    def expand(self, tensor: OptArray) -> OptArray:
        shape = list(tensor.shape)
        return tensor.unsqueeze(0).expand([self.n_terms] + shape)

    @staticmethod
    def reduce(tensors: OptArray) -> OptArray:
        return opt_sum(tensors, 0)

    @staticmethod
    def transforms(tensors: OptArray, op_dict: dict[int, OptArray]) -> OptArray:
        order = tensors.ndim - 1
        n = len(op_dict)
        op_ax = order + n
        assert op_ax < MAX_EINSUM_AXES

        ax_list = list(sorted(op_dict.keys(), key=(lambda ax: tensors.shape[ax + 1])))
        mat_list = [op_dict[ax] for ax in ax_list]

        args = [(tensors, [op_ax] + list(range(order)))]
        args.extend((mat_list[i], [op_ax, order + i, ax_list[i]]) for i in range(n))
        ans_axes = [op_ax] + [order + ax_list.index(ax) if ax in ax_list else ax for ax in range(order)]
        args.append((ans_axes,))

        ans = opt_einsum(*chain(*args))
        return ans

    @staticmethod
    def traces(tensors1: OptArray, tensors2: OptArray, ax: int) -> OptArray:
        assert tensors1.shape == tensors2.shape
        order = tensors1.ndim - 1
        assert ax < order
        assert order + 2 < MAX_EINSUM_AXES

        op_ax = order
        i_ax = order + 1
        j_ax = order + 2

        axes1 = list(range(order))
        axes1[ax] = i_ax
        axes2 = list(range(order))
        axes2[ax] = j_ax
        return opt_einsum(tensors1, [op_ax] + axes1, tensors2, [op_ax] + axes2, [op_ax, i_ax, j_ax])


class MasterEqn(object):
    r"""Solve the equation wrt Sum-of-Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self, op: SumProdOp, state: CannonialModel) -> None:
        assert op.ends == state.ends
        self.op = op
        self.state = state

        self.mean_fields = dict()  # type: dict[Tuple[Node, int], OptArray]
        self.densities = dict()  # type: dict[Node, OptArray]
        # primitive
        dual = state.frame.dual
        for q in state.ends:
            p, i = dual(q, 0)
            self.mean_fields[(p, i)] = self.op[q]
        self._node_visitor = state.frame.node_visitor(start=state.root, method='BFS')
        return

    def calculate(self, for_all: bool = True) -> None:
        self._get_mean_fields_type1()
        if for_all:
            self._get_mean_fields_type2()
            self._get_densities()
        return

    def node_eom(self, node: Node) -> Callable[[OptArray], OptArray]:
        ax = self.state.axes[node]

        order = self.state.frame.order(node)
        op_list = {i: self.mean_fields[node, i] for i in range(order)}
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce
        trace = self.op.traces

        def _dd(a: OptArray) -> OptArray:
            assert a.ndim == order
            a = expand(a)
            tmp = transforms(a, op_list)
            if ax is not None:
                # Projection
                projection = transforms(a, {ax: trace(tmp, a.conj(), ax)})
                tmp -= projection
            ans = reduce(tmp)
            if ax is not None:
                # Inversion
                den = self.densities[node]
                ans = opt_tensordot(ans, den.pinverse(), ([ax], [1]))
                ans = ans.moveaxis(-1, ax)
            return ans

        return _dd

    def node_eom_op(self, node: Node) -> OptArray:
        if node is not self.state.root:
            raise NotImplementedError

        a = self.state[node]
        dims = a.shape
        order = a.ndim

        ax_list = list(sorted(range(order), key=(lambda ax: dims[ax])))
        mat_list = [self.mean_field(node, ax) for ax in ax_list]

        op_ax = 2 * order

        from_axes = list(range(order))
        to_axes = list(range(order, 2 * order))

        args = [(mat_list[i], [op_ax, order + ax_list[i], ax_list[i]]) for i in range(order)]
        args.append((to_axes + from_axes,))
        diff = opt_einsum(*chain(*args))

        return diff

    def _mean_field_with_node(self, p: Node, i: int) -> OptArray:
        order = self.state.frame.order(p)

        a = self.op.expand(self.state[p])
        conj_a = a.conj()
        op_list = {_i: self.mean_fields[(p, _i)] for _i in range(order) if _i != i}
        return self.op.traces(conj_a, self.op.transforms(a, op_list), ax=i)

    def _get_mean_fields_type1(self) -> None:
        """From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual
        mf = self._mean_field_with_node

        for q in reversed(self._node_visitor):
            j = axes[q]
            if j is not None:
                p, i = dual(q, j)
                self.mean_fields[(p, i)] = mf(q, j)
        return

    def _get_mean_fields_type2(self) -> None:
        """From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual
        mf = self._mean_field_with_node

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                self.mean_fields[(p, i)] = mf(q, j)
        return

    def _get_densities(self) -> None:
        state = self.state
        axes = state.axes
        dual = state.frame.dual
        order = state.frame.order

        for p in self._node_visitor:
            i = axes[p]
            if i is None:
                continue
            q, j = dual(p, i)
            k = axes[q]
            a_q = state[q]
            if k is not None:
                den_q = self.densities[q]
                a_q = opt_tensordot(den_q, a_q, ([1], [k]))
            ops = [_j for _j in range(order(q)) if _j != j]
            ans = opt_tensordot(a_q.conj(), a_q, (ops, ops))
            self.densities[p] = ans
        return


class Propagator:
    tol = backend.ODE_TOL
    max_steps = 1_000_000_000

    def __init__(self,
                 op: SumProdOp,
                 state: CannonialModel,
                 dt: float,
                 ode_method: str = 'dopri5',
                 ps_method: int = 0) -> None:

        self.eom = MasterEqn(op, state)
        self.state = self.eom.state
        self.op = self.eom.op

        self.dt = dt
        self.ode_method = ode_method
        self.ps_method = ps_method

        self._node_visitor = self.state.frame.node_visitor(self.state.root)
        self._node_link_visitor = self.state.frame.node_link_visitor(self.state.root)
        self._depths = self.state.depths
        self._move1 = self.state.split_unite_move
        self._move2 = self.state.unite_split_move
        return

    def __iter__(self) -> Generator[float, None, None]:
        for i in range(self.max_steps):
            yield (self.dt * i)
            self.step()
        return

    def step(self) -> None:
        if self.ps_method == 0:
            self.eom.calculate(for_all=True)
            for p in self._node_visitor:
                self._node_step(p, 1.0)
        elif self.ps_method == 1:
            self.eom.calculate(for_all=False)
            self.ps1_forward_step(0.5)
            self.ps1_backward_step(0.5)
        elif self.ps_method == 2:
            self.eom.calculate(for_all=False)
            self.ps2_forward_step(0.5)
            self.ps2_backward_step(0.5)
        else:
            raise NotImplementedError
        return

    def _node_step(self, p: Node, ratio: float) -> None:
        ans = self._odeint(self.eom.node_eom(p), self.state[p], ratio)
        self.state.opt_update(p, ans)
        return

    def ps1_forward_step(self, ratio: float) -> None:
        move = self._move1
        depths = self._depths

        for p, i, q, j in self._node_link_visitor:
            assert p is self.state.root
            if depths[p] < depths[q]:
                move(i, op=self._ps1_mid_op(p, i, q, j, None))
            else:
                self._node_step(p, ratio)
                move(i, op=self._ps1_mid_op(p, i, q, j, -ratio))
        self._node_step(self.state.root, ratio)
        return

    def ps1_backward_step(self, ratio: float) -> None:
        move = self._move1
        depths = self._depths

        self._node_step(self.state.root, ratio)
        for q, j, p, i in reversed(self._node_link_visitor):
            assert p is self.state.root
            if depths[p] < depths[q]:
                move(i, op=self._ps1_mid_op(p, i, q, j, -ratio))
                self._node_step(q, ratio)
            else:
                move(i, op=self._ps1_mid_op(p, i, q, j, None))
        return

    def _ps1_mid_op(self, p: Node, i: int, q: Node, j: int, ratio: Optional[float]) -> Callable[[OptArray], OptArray]:
        """Assue the tensor for p in self.state has been updated."""

        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce

        def _op(mid: OptArray) -> OptArray:
            l_mat = self.eom._mean_field_with_node(p, i)
            r_mat = self.eom.mean_fields[p, i]
            self.eom.mean_fields[q, j] = l_mat
            del self.eom.mean_fields[p, i]
            if ratio is None:
                return mid
            else:
                _dd = lambda a: reduce(transforms(expand(a), {0: l_mat, 1: r_mat}))
                return self._odeint(_dd, mid, ratio)

        return _op

    def ps2_forward_step(self, ratio: float) -> None:
        depths = self._depths
        move2 = self._move2
        move1 = self._move1
        tol = self.tol

        it = self._node_link_visitor
        end = len(it) - 1
        for n, (p, i, q, j) in enumerate(it):
            # print(p, i, q, j)
            assert p is self.state.root
            if depths[p] < depths[q]:
                move1(i, op=self._ps1_mid_op(p, i, q, j, None))
            else:
                move2(i, op=self._ps2_mid_op(p, i, q, j, ratio), tol=tol)
                self.eom.mean_fields[q, j] = self.eom._mean_field_with_node(p, i)
                del self.eom.mean_fields[p, i]
                if n != end:
                    self._node_step(q, -ratio)
        return

    def ps2_backward_step(self, ratio: float) -> None:
        depths = self._depths
        move2 = self._move2
        move1 = self._move1
        tol = self.tol

        it = reversed(self._node_link_visitor)
        for n, (q, j, p, i) in enumerate(it):
            assert p is self.state.root
            # print(p, i, q, j)
            if depths[p] < depths[q]:
                if n != 0:
                    self._node_step(p, -ratio)
                move2(i, op=self._ps2_mid_op(p, i, q, j, ratio), tol=tol)
                self.eom.mean_fields[q, j] = self.eom._mean_field_with_node(p, i)
                del self.eom.mean_fields[p, i]
            else:
                move1(i, op=self._ps1_mid_op(p, i, q, j, None))
        return

    def _ps2_mid_op(self, p: Node, i: int, q: Node, j: int, ratio: float) -> Callable[[OptArray], OptArray]:
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce

        def _op(mid: OptArray) -> OptArray:
            ord_p = self.state.frame.order(p)
            ord_q = self.state.frame.order(q)
            l_ops = [self.eom.mean_fields[p, _i] for _i in range(ord_p) if _i != i]
            r_ops = [self.eom.mean_fields[q, _j] for _j in range(ord_q) if _j != j]
            op_dict = dict(enumerate(l_ops + r_ops))

            _dd = lambda a: reduce(transforms(expand(a), op_dict))
            return self._odeint(_dd, mid, ratio)

        return _op

    def _odeint(self, func: Callable[[OptArray], OptArray], y0: OptArray, ratio: float):
        return odeint(func, y0, ratio * self.dt, method=self.ode_method)
