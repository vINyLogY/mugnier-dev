# coding: utf-8

from math import prod, inf
from matplotlib.pyplot import step
import numpy as np
import torch
from tqdm import tqdm, trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT
from mugnier.heom.bath import BoseEinstein, Drude
from mugnier.operator.spo import MasterEqn, Propagator


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
    dim = 20
    dims = [dim] * corr.k_max
    heom_op = Hierachy(h, op, corr, dims)
    s = TensorTrainEDT(rdo, dims, rank=19)
    solver = MasterEqn(heom_op, s)
    solver.calculate()

    # for n in solver.state.frame.node_visitor(start=s.root):

    #     func = solver.node_eom(n)
    #     diff_norm = np.linalg.norm(func(s[n]).cpu().numpy())
    #     print(n, diff_norm)

    # for (n, i), v in solver.mean_fields.items():
    #     print(n, i, s.shape(n), v.shape)
    # np.testing.assert_almost_equal(0.01611424664290165, diff_norm)

    # Propagator settings:
    steps = 1000
    interval = __(0.01, 'fs')

    logger1 = Logger(filename=f'tdvp_heom_{corr.k_max}({dim})-dt_{interval}-{backend.device}.log', level='info').logger
    logger1.info('# time_(au) rdo00 rdo01 rdo10 rdo11')
    propagator = Propagator(heom_op, s, interval.au, ps_method=0)
    it = trange(steps)
    for n in it:
        propagator.step()
        _t = n * interval.au
        rdo = s.get_rdo()
        trace = rdo[0, 0] + rdo[1, 1]
        logger1.info(f'{_t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')
        it.set_description(f'tr:{trace}')


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    test_hierachy()
