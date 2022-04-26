# coding: utf-8
from inspect import trace
import numpy as np
import torch
from tqdm import tqdm
from mugnier.libs.backend import array
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.structure.frame import MultiLayerMCTDH, Singleton, TensorTrain
from mugnier.structure.network import Node, End, Point, State
from mugnier.structure.operator import SumProdOp
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy
from mugnier.heom.bath import Correlation, BoseEinstein, Drude
from mugnier.structure.propagator import SumProdEOM


def test_hierachy():
    spd = Drude(__(200, '/cm').au, __(50, '/cm').au)
    be = BoseEinstein(n=0, beta=__(1 / 100_000, '/K').au)
    corr = Correlation(spd, be)
    corr.print()
    s = ExtendedDensityTensor(corr.k_max)
    dims = [400] * corr.k_max

    e = __(5000.0, '/cm').au
    v = __(0.0, '/cm').au

    h = array([[0, v], [v, e]])
    op = array([[0.0, 0.0], [0.0, 1.0]])
    rdo = array([[0.5, 0.5], [0.5, 0.5]])

    s.initialize(rdo, dims)

    heom_op = Hierachy(h, op, corr, dims)

    solver = SumProdEOM(heom_op, s)

    steps = 1000
    interval = __(0.1, 'fs').au
    callback_interval = 1

    logger1 = Logger(filename='gpu.log', level='info').logger
    logger1.info('# time rdo00 rdo01 rdo10 rdo11')

    with tqdm(total=steps) as pbar:
        for n, (_t, _s) in enumerate(solver.propagator(steps=steps, interval=interval)):
            if n % callback_interval == 0:

                time = __(_t).convert_to('fs').value
                rdo = _s[_s.root]
                trace = rdo[0, 0, 0] + rdo[1, 1, 0]
                logger1.info(f'{time} {rdo[0, 0, 0]} {rdo[0, 1, 0]} {rdo[1, 0, 0]} {rdo[1, 1, 0]}')
                pbar.set_description(f'Time: {time:.2f} fs; Tr: {trace}')
                pbar.update(callback_interval)


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    test_hierachy()
