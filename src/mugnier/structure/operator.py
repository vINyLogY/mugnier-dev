# coding: utf-8

from functools import partial
from typing import Iterable, Tuple


from mugnier.structure.network import Node, Edge, State
from mugnier.libs.backend import Array, asarray


class ProdOp(object):
    def __init__(self, valuation: dict[Edge, Array]) -> None:
        self._valuation = valuation
        return

    def eom(self, state: State, node: Node):
        ans = {e: self.mean_field(state)[node, e] for e in state.frame.near_node(node)}
        if node is state.root:
            return ans
        else:
            raise NotImplementedError

    def mean_fields(self, state: State) -> dict[Tuple[Node, Edge], Array]:
        def snd(x): return x[1]
        iters = [n for n in sorted(state.depthes.items(), key=snd, reverse=True)]

    def reduced_densities(self, state: State) -> dict[Node, Array]:
        raise NotImplementedError


class SumProdOp(object):
    def __init__(self,  valuations: list[dict[Edge, Array]]) -> None:
        self.ops = [ProdOp(v) for v in valuations]
        return
