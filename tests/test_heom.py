# coding: utf-8

import numpy as np
import torch
from tqdm import tqdm
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy
from mugnier.heom.bath import Correlation, BoseEinstein, Drude
from mugnier.operator.spo import Integrator


def test_hierachy():
    # System settings:
    e = __(5000.0, '/cm').au
    v = __(0.0, '/cm').au
    h = backend.array([[0, v], [v, e]])
    op = backend.array([[0.0, 0.0], [0.0, 1.0]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=0, beta=__(1 / 100_000, '/K').au)
    corr = Drude(__(200, '/cm').au, __(50, '/cm').au, distr)

    # HEOM settings:
    dim = 400
    dims = [dim] * corr.k_max
    heom_op = Hierachy(h, op, corr, dims)
    s = ExtendedDensityTensor(rdo, dims)
    solver = Integrator(heom_op, s)

    func = solver.single_diff(s.root)
    diff_norm = np.linalg.norm(func(s[s.root]).to('cpu').numpy())
    np.testing.assert_almost_equal(0.01611424664290165, diff_norm)

    # # Propagator settings:
    # callback_interval = 100
    # steps = 1000 * callback_interval
    # interval = __(0.1 / callback_interval, 'fs').au

    # logger1 = Logger(filename=f'dim_{dim}-dt_{interval:.4f}-{backend.device}.log', level='info').logger
    # logger1.info('# time rdo00 rdo01 rdo10 rdo11')
    # with tqdm(total=steps) as pbar:
    #     for n, (_t, _s) in enumerate(solver.propagator(steps=steps, interval=interval)):
    #         if n % callback_interval == 0:
    #             time = __(_t).convert_to('fs').value
    #             rdo = _s[_s.root].reshape((2, 2, -1))[:, :, 0]
    #             trace = rdo[0, 0] + rdo[1, 1]
    #             logger1.info(f'{_t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')
    #             pbar.set_description(f'Time: {time:.2f} fs; Tr: {trace}')
    #             pbar.update(callback_interval)


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    test_hierachy()
