# coding: utf-8

from itertools import chain
from math import prod
from typing import Callable, Generator, Literal, Optional, Tuple

import numpy as np
from mugnier.libs.backend import (MAX_EINSUM_AXES, Array, OptArray, eye, odeint, opt_einsum, opt_inv, opt_pinv,
                                  opt_regularized_qr, opt_sum, opt_svd, opt_tensordot, optimize, opt_cat, opt_split)
from mugnier.state.frame import End, Node
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

        args = list(chain(*args))
        ans = opt_einsum(*args)
        return ans

    @staticmethod
    def traces(tensors1: OptArray, tensors2: OptArray, ax: int) -> OptArray:
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
    REG_TYPE = True  # type: Literal[1, 2]

    @staticmethod
    def transform(tensor: OptArray, op_dict: dict[int, OptArray]) -> OptArray:
        order = tensor.ndim
        n = len(op_dict)
        assert order + n - 1 < MAX_EINSUM_AXES

        ax_list = list(sorted(op_dict.keys(), key=(lambda ax: tensor.shape[ax])))
        # ax_list = list(sorted(op_dict.keys()))
        mat_list = [op_dict[ax] for ax in ax_list]

        args = [(tensor, list(range(order)))]
        args.extend((mat_list[i], [order + i, ax_list[i]]) for i in range(n))
        ans_axes = [order + ax_list.index(ax) if ax in ax_list else ax for ax in range(order)]
        args.append((ans_axes,))

        ans = opt_einsum(*chain(*args))
        return ans

    @staticmethod
    def trace(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
        order = tensor1.ndim
        assert ax < order
        assert order + 1 < MAX_EINSUM_AXES

        i_ax = order
        j_ax = order + 1

        axes1 = list(range(order))
        axes1[ax] = i_ax
        axes2 = list(range(order))
        axes2[ax] = j_ax
        return opt_einsum(tensor1, axes1, tensor2, axes2, [i_ax, j_ax])

    def __init__(self, op: SumProdOp, state: CannonialModel) -> None:
        assert op.ends == state.ends
        self.op = op
        self.state = state

        self.mean_fields = dict()  # type: dict[Tuple[Node, int], OptArray]
        self.densities = dict()  # type: dict[Node, OptArray]

        # Temp for regularization
        self._reg = dict()  # type: dict[Node, OptArray]
        self._reg_r = dict()  # type: dict[Node, OptArray]
        self._reg_s = dict()  # type: dict[Node, OptArray]
        self._reg_v = dict()  # type: dict[Node, OptArray]
        # primitive
        dual = state.frame.dual
        for q in state.ends:
            p, i = dual(q, 0)
            self.mean_fields[(p, i)] = self.op[q]
        self._node_visitor = state.frame.node_visitor(start=state.root, method='BFS')
        self._shape_list = [state.shape(p) for p in self._node_visitor]
        self._size_list = [prod(s) for s in self._shape_list]
        return

    def vectorize(self, tensors: list[OptArray]) -> OptArray:
        return opt_cat([a.flatten() for a in tensors])

    def vector_split(self, vec: OptArray) -> list[OptArray]:
        tensors = opt_split(vec, self._size_list)
        return [a.reshape(s) for a, s in zip(tensors, self._shape_list)]

    def vector(self) -> OptArray:
        return self.vectorize([self.state[p] for p in self._node_visitor])

    def vector_update(self, vec: OptArray) -> None:
        update = self.state.opt_update
        for p, a in zip(self._node_visitor, self.vector_split(vec)):
            update(p, a)
        return

    def vector_reg_eom(self, fast: bool = False, use_qr=False) -> Callable[[OptArray], OptArray]:
        axes = self.state.axes
        order_of = self.state.frame.order
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce

        transform = self.transform
        trace = self.trace
        update = self.vector_update
        split = self.vector_split
        vectorize = self.vectorize
        visitor = self._node_visitor
        get_mf = self._get_mean_fields_type1
        if use_qr:
            get_reg = self._get_fast_reg_mean_fields_qr if fast else self._get_reg_mean_fields_qr
        else:
            get_reg = self._get_fast_reg_mean_fields if fast else self._get_reg_mean_fields

        def _dd(a: OptArray) -> OptArray:
            ans_list = []
            update(a)
            get_mf()
            get_reg()

            for p, a in zip(visitor, split(a)):
                order = order_of(p)
                ax = axes[p]
                assert a.ndim == order

                op_list = {i: self.mean_fields[p, i] for i in range(order) if i != ax}
                if ax is not None:
                    # Inversion
                    if use_qr:
                        op_list[ax] = opt_inv(self._reg_r[p]) @ self._reg[p]
                    else:
                        op_list[ax] = self._reg_v[p].mH @ (1.0 / self._reg_s[p]).diag() @ self._reg[p]

                ans = reduce(transforms(expand(a), op_list))
                if ax is not None:
                    # Projection
                    projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                    ans -= projection

                ans_list.append(ans)
            return vectorize(ans_list)

        return _dd

    def vector_eom(self) -> Callable[[OptArray], OptArray]:
        axes = self.state.axes
        order_of = self.state.frame.order
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce

        transform = self.transform
        trace = self.trace

        def _dd(a: OptArray) -> OptArray:
            ans_list = []
            self.vector_update(a)
            self._get_mean_fields_type1()
            self._get_mean_fields_type2()
            self._get_densities()
            for p, a in zip(self._node_visitor, self.vector_split(a)):
                order = order_of(p)
                ax = axes[p]
                assert a.ndim == order

                op_list = {i: self.mean_fields[p, i] for i in range(order)}
                if ax is not None:
                    # Inversion
                    op_list[ax] = opt_pinv(self.densities[p]) @ op_list[ax]
                ans = reduce(transforms(expand(a), op_list))
                if ax is not None:
                    # Projection
                    projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                    ans -= projection

                ans_list.append(ans)
            return self.vectorize(ans_list)

        return _dd

    def node_reg_eom(self, node: Node) -> Callable[[OptArray], OptArray]:
        ax = self.state.axes[node]
        order = self.state.frame.order(node)
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce
        transform = self.transform
        trace = self.trace

        op_list = {i: self.mean_fields[node, i] for i in range(order) if i != ax}
        if ax is not None:
            # Inversion
            op_list[ax] = self._reg_v[node].mH @ (1.0 / self._reg_s[node]).diag() @ self._reg[node]

        def _dd(a: OptArray) -> OptArray:
            assert a.ndim == order
            ans = reduce(transforms(expand(a), op_list))
            if ax is not None:
                # Projection
                projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                ans -= projection
            return ans

        return _dd

    def node_eom(self, node: Node) -> Callable[[OptArray], OptArray]:
        ax = self.state.axes[node]
        order = self.state.frame.order(node)
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce

        transform = self.transform
        trace = self.trace

        def _dd(a: OptArray) -> OptArray:
            assert a.ndim == order
            op_list = {i: self.mean_fields[node, i] for i in range(order)}
            if ax is not None:
                # Inversion
                op_list[ax] = opt_pinv(self.densities[node]) @ op_list[ax]
            ans = reduce(transforms(expand(a), op_list))
            if ax is not None:
                # Projection
                projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                ans -= projection
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

    def _get_fast_reg_mean_fields(self) -> None:
        """From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual

        def regularize(p: Node, i: int) -> Tuple[OptArray, ...]:
            order = self.state.frame.order(p)
            shape = list(self.state.shape(p))
            dim = shape.pop(i)
            a = self.state[p]
            u, s, v = opt_svd(a.moveaxis(i, -1).reshape((-1, dim)))
            a = self.op.expand(a)
            u = self.op.expand(u.reshape(shape + [-1]).moveaxis(-1, i))
            conj_a = a.conj()
            conj_u = u.conj()
            op_list = {_i: self.mean_fields[(p, _i)] for _i in range(order) if _i != i}
            trans = self.op.transforms(a, op_list)
            mf = self.op.traces(conj_a, trans, ax=i)
            reg = self.op.traces(conj_u, trans, ax=i)

            return mf, reg, s, v

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                mf, reg, s, v = regularize(q, j)
                self.mean_fields[(p, i)] = mf
                self._reg[p] = reg
                self._reg_s[p] = s
                self._reg_v[p] = v
        return

    def _get_fast_reg_mean_fields_qr(self) -> None:
        """From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual

        def regularize(p: Node, i: int) -> Tuple[OptArray, ...]:
            order = self.state.frame.order(p)
            shape = list(self.state.shape(p))
            dim = shape.pop(i)
            a = self.state[p]
            u, r = opt_regularized_qr(a.moveaxis(i, -1).reshape((-1, dim)))
            a = self.op.expand(a)
            u = self.op.expand(u.reshape(shape + [-1]).moveaxis(-1, i))
            conj_a = a.conj()
            conj_u = u.conj()
            op_list = {_i: self.mean_fields[(p, _i)] for _i in range(order) if _i != i}
            trans = self.op.transforms(a, op_list)
            mf = self.op.traces(conj_a, trans, ax=i)
            reg = self.op.traces(conj_u, trans, ax=i)

            return mf, reg, r

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                mf, reg, r = regularize(q, j)
                # mf, reg, s, v = regularize(q, j)
                self.mean_fields[(p, i)] = mf
                self._reg[p] = reg
                self._reg_r[p] = r
        return

    def _get_reg_mean_fields(self) -> None:
        """From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual

        def regularize(p: Node, i: int) -> Tuple[OptArray, ...]:
            order = self.state.frame.order(p)
            ax = axes[p]
            _a = self.state[p]
            if ax is not None:
                _a = self.transform(_a, {ax: self._reg_s[p].diag() @ self._reg_v[p]})
            shape = list(_a.shape)
            dim = shape.pop(i)
            u, s, v = opt_svd(_a.moveaxis(i, -1).reshape((-1, dim)))
            u = u.reshape(shape + [-1]).moveaxis(-1, i)

            op_list = {_i: self.mean_fields[(p, _i)] for _i in range(order) if _i != i and _i != ax}
            if ax is not None:
                op_list[ax] = self._reg[p]

            a = self.op.expand(self.state[p])
            conj_u = self.op.expand(u.conj())
            reg = self.op.traces(conj_u, self.op.transforms(a, op_list), ax=i)

            return reg, s, v

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                reg, s, v = regularize(q, j)
                self._reg[p] = reg
                self._reg_s[p] = s
                self._reg_v[p] = v
        return

    def _get_reg_mean_fields_qr(self) -> None:
        """Use QR instead of SVD. From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual

        def regularize(p: Node, i: int) -> Tuple[OptArray, ...]:
            order = self.state.frame.order(p)
            ax = axes[p]
            _a = self.state[p]
            if ax is not None:
                _a = self.transform(_a, {ax: self._reg_r[p]})
            shape = list(_a.shape)
            dim = shape.pop(i)
            u, r = opt_regularized_qr(_a.moveaxis(i, -1).reshape((-1, dim)))
            u = u.reshape(shape + [-1]).moveaxis(-1, i)

            op_list = {_i: self.mean_fields[(p, _i)] for _i in range(order) if _i != i and _i != ax}
            if ax is not None:
                op_list[ax] = self._reg[p]

            a = self.op.expand(self.state[p])
            conj_u = self.op.expand(u.conj())
            reg = self.op.traces(conj_u, self.op.transforms(a, op_list), ax=i)

            return reg, r

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                reg, r = regularize(q, j)
                self._reg[p] = reg
                self._reg_r[p] = r
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
    max_steps = 1_000_000_000

    def __init__(self,
                 op: SumProdOp,
                 state: CannonialModel,
                 dt: float,
                 ode_method: Literal['bosh3', 'dopri5', 'dopri8'] = 'dopri5',
                 reg_method: Literal['proper', 'fast', 'proper_qr', 'fast_qr', 'pinv'] = 'proper',
                 ps_method: Literal['cmf', 'ps1', 'ps2', 'vmf'] = 'vmf') -> None:

        self.eom = MasterEqn(op, state)
        self.state = self.eom.state
        self.op = self.eom.op

        self.dt = dt
        self.ode_method = ode_method
        self.ps_method = ps_method
        self.reg_method = reg_method

        self._node_visitor = self.state.frame.node_visitor(self.state.root)
        self._node_link_visitor = self.state.frame.node_link_visitor(self.state.root)
        self._depths = self.state.depths
        self._move1 = self.state.split_unite_move
        self._move2 = self.state.unite_split_move

        self.ode_step_counter = []
        return

    def __iter__(self) -> Generator[float, None, None]:
        for i in range(1, self.max_steps):
            self.step()
            yield (self.dt * i)
        return

    def step(self) -> None:
        if self.ps_method == 'cmf':
            self.eom._get_mean_fields_type1()
            self.eom._get_reg_mean_fields()
            for p in self._node_visitor:
                self._node_step(p, 1.0)
        elif self.ps_method == 'ps1':
            self.eom._get_mean_fields_type1()
            self.ps1_forward_step(0.5)
            self.ps1_backward_step(0.5)
        elif self.ps_method == 'ps2':
            self.eom._get_mean_fields_type1()
            self.ps2_forward_step(0.5)
            self.ps2_backward_step(0.5)
        elif self.ps_method == 'vmf':
            y = self.eom.vector()
            if self.reg_method == 'fast':
                ans = self._odeint(self.eom.vector_reg_eom(fast=True, use_qr=False), y, 1.0)
            elif self.reg_method == 'proper':
                ans = self._odeint(self.eom.vector_reg_eom(fast=False, use_qr=False), y, 1.0)
            elif self.reg_method == 'fast_qr':
                ans = self._odeint(self.eom.vector_reg_eom(fast=True, use_qr=True), y, 1.0)
            elif self.reg_method == 'proper_qr':
                ans = self._odeint(self.eom.vector_reg_eom(fast=False, use_qr=True), y, 1.0)
            elif self.reg_method == 'pinv':
                ans = self._odeint(self.eom.vector_eom(), y, 1.0)
            else:
                raise NotImplementedError(f'No regularization method `{self.reg_method}`.')
            self.eom.vector_update(ans)
        else:
            raise NotImplementedError(f'No Projector splitting method `{self.ps_method}`.')
        return

    def _node_step(self, p: Node, ratio: float) -> None:
        ans = self._odeint(self.eom.node_reg_eom(p), self.state[p], ratio)
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

        it = self._node_link_visitor
        end = len(it) - 1
        for n, (p, i, q, j) in enumerate(it):
            assert p is self.state.root
            if depths[p] < depths[q]:
                move1(i, op=self._ps1_mid_op(p, i, q, j, None))
            else:
                move2(i, op=self._ps2_mid_op(p, i, q, j, ratio))
                self.eom.mean_fields[q, j] = self.eom._mean_field_with_node(p, i)
                del self.eom.mean_fields[p, i]
                if n != end:
                    self._node_step(q, -ratio)
        return

    def ps2_backward_step(self, ratio: float) -> None:
        depths = self._depths
        move2 = self._move2
        move1 = self._move1

        it = reversed(self._node_link_visitor)
        for n, (q, j, p, i) in enumerate(it):
            assert p is self.state.root
            if depths[p] < depths[q]:
                if n != 0:
                    self._node_step(p, -ratio)
                move2(i, op=self._ps2_mid_op(p, i, q, j, ratio))
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

    def _odeint(self, func: Callable[[OptArray], OptArray], y0: OptArray, ratio: float) -> OptArray:
        ans, n_eval = odeint(func, y0, ratio * self.dt, method=self.ode_method)
        self.ode_step_counter.append(n_eval)
        return ans
