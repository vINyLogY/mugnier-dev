from math import prod
from typing import Optional, Tuple

from mugnier.libs.backend import (Array, OptArray, array, eye, np, optimize, zeros)
from mugnier.state.frame import End, Frame, Node, Point


class Model:
    """Network is a Frame with valuation for each node.
    """

    def __init__(self, frame: Frame) -> None:
        """
        Args:
            frame: Topology of the tensor network;
        """
        self.frame = frame
        self.ends = frame.ends
        self._dims = dict()  # type: dict[Tuple[Point, int], int]
        self._valuation = dict()  # type: dict[Node, OptArray]
        return

    def shape(self, p: Point) -> list[Optional[int]]:
        assert p in self.frame.world

        if isinstance(p, End):
            dim = self._dims.get((p, 0))
            node_shape = [dim, dim]
        else:
            # isinstance(p, Node)
            node_shape = [self._dims.get((p, i)) for i in range(self.frame.order(p))]
        return node_shape

    def __getitem__(self, p: Node) -> OptArray:
        return self._valuation[p]

    def __setitem__(self, p: Node, array: Array) -> None:
        assert p in self.frame.nodes
        order = self.frame.order(p)
        assert array.ndim == order

        # Check confliction
        for i in range(order):
            for pair in [(p, i), self.frame.dual(p, i)]:
                _dim = array.shape[i]
                if pair in self._dims:
                    assert self._dims[pair] == _dim
                else:
                    self._dims[pair] = _dim

        self._valuation[p] = optimize(array)
        return

    def __delitem__(self, p: Node) -> None:
        for i in range(self.frame.order(p)):
            del self._dims[(p, i)]
        del self._valuation[p]
        return

    def opt_update(self, p: Node, array: OptArray) -> None:
        assert p in self._valuation
        self._valuation[p] = array

    def fill_zeros(self, dims: Optional[dict[Tuple[Node, int], int]] = None, default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the dimension for each Edge in dims (default is 1).
        """
        if dims is None:
            dims = dict()

        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        _dims = self._dims

        for p in self.frame.nodes:
            if p not in valuation:
                shape = [
                    _dims.get((p, i), dims.get((p, i), dims.get(dual(p, i), default_dim))) for i in range(order(p))
                ]
                self[p] = zeros(shape)

        return


class CannonialModel(Model):
    """
    State is a Network with Tree-shape and root.
    """

    def __init__(self, frame: Frame, root: Node) -> None:
        super().__init__(frame)

        self._root = root
        self._axes = None  # type: Optional[dict[Node, Optional[int]]]
        return

    @property
    def root(self) -> Node:
        return self._root

    @root.setter
    def root(self, value: Node) -> None:
        assert value in self.frame.nodes
        self._root = value
        self._axes = None
        return

    @property
    def axes(self) -> dict[Node, Optional[int]]:
        if self._axes is None:
            ans = {self.root: None}
            for m, _, n, j in self.frame.node_link_visitor(self.root):
                if m in ans and n not in ans:
                    ans[n] = j
            self._axes = ans
        else:
            ans = self._axes
        return ans

    def fill_eyes(self, dims: Optional[dict[Tuple[Node, int], int]] = None, default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the each dimension tensors (default is 1).
        """
        if dims is None:
            dims = dict()
        nodes = self.frame.nodes
        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        saved_dims = self._dims
        axes = self.axes

        for p in nodes:
            if p not in valuation:
                ax = axes[p]
                shape = [
                    saved_dims.get((p, i), dims.get((p, i), dims.get(dual(p, i), default_dim)))
                    for i in range(order(p))
                ]
                if ax is None:
                    ans = zeros((prod(shape),))
                    ans[0] = 1.0
                    ans = ans.reshape(shape)
                else:
                    _m = shape.pop(ax)
                    _n = prod(shape)
                    ans = np.moveaxis(eye(_m, _n).reshape([_m] + shape), 0, ax)
                self[p] = ans
        return
