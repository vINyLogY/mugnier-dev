# coding: utf-8

from math import prod, inf
from matplotlib.pyplot import step
import numpy as np
from sympy import root
import torch
from tqdm import tqdm, trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, Layer2EDT, Layer3EDT, TensorTrainEDT
from mugnier.heom.bath import BoseEinstein, Drude
from mugnier.operator.spo import MasterEqn, Propagator
from mugnier.state.frame import End


def test_hierachy():
    # System settings:
    e = __(5000.0, '/cm').au
    v = __(0.0, '/cm').au
    h = backend.array([[0, v], [v, e]])
    op = backend.array([[0.0, 0.0], [0.0, 1.0]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=3, beta=__(1 / 300, '/K').au)
    corr = Drude(__(500, '/cm').au, __(50, '/cm').au, distr)

    # HEOM settings:
    dim = 40
    rank = 5
    dims = [dim] * corr.k_max
    heom_op = Hierachy(h, op, corr, dims)
    s = Layer2EDT(rdo, dims, rank=rank)

    # Propagator settings:
    steps = 1000
    interval = __(0.1, 'fs')
    callback_steps = 1
    ps_method = None

    propagator = Propagator(heom_op, s, interval.au, ps_method=ps_method)
    logger1 = Logger(filename=f'MCTDH2_ps{ps_method}_{corr.k_max}({dim})[{rank}]-dt_{interval}-{backend.device}.log',
                     level='info').logger
    logger1.info('# time_(fs) rdo00 rdo01 rdo10 rdo11')
    it = trange(steps)
    for n, _t in zip(it, propagator):
        rdo = s.get_rdo()
        trace = rdo[0, 0] + rdo[1, 1]
        logger1.info(f'{_t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')

        if n % callback_steps == 0:
            rank = [
                d for (p, i), d in s._dims.items()
                if not isinstance(p, End) and not isinstance(s.frame.dual(p, i)[0], End)
            ]
            vals = {p: torch.max(torch.abs(s[p])) for p in s.frame.nodes}
            it.set_description(f'Tr:{trace} | Coh:{abs(rdo[0, 1])} | rank:{max(rank)} | mv:{max(vals.values())}')


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    test_hierachy()
