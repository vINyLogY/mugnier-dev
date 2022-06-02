# coding: utf-8

import enum
from math import log, log10, prod
from time import time
from tqdm import trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT, TensorTreeEDT
from mugnier.heom.bath import BoseEinstein, Correlation, Drude, UnderdampedBrownian
from mugnier.operator.spo import MasterEqn, Propagator
from mugnier.state.frame import End


def test_hierachy(freq_max: float = 2000,
                  re: float = 1000,
                  width: float = 50,
                  dof: int = 4,
                  n_ltc: int = 1,
                  dim: int = 10,
                  rank: int = 20,
                  dry_run: bool = False):
    # System settings:
    SCALE = 1 / __(5000.0, '/cm').au
    e = __(5000.0 * SCALE, '/cm').au
    v = __(0.0 * SCALE, '/cm').au
    pop = 0.5
    h = backend.array([[-pop * e, v], [v, (1.0 - pop) * e]])
    op = backend.array([[-pop, 0.0], [0.0, 1.0 - pop]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=n_ltc, beta=__(1 / SCALE / 300, '/K').au)
    distr.decomposition_method = 'Pade'
    corr = Correlation(distr)

    freq_space = [freq_max / (dof + 1) * (n + 1) for n in range(dof)]
    bosons = []  # type:list[UnderdampedBrownian]
    for _n, freq in enumerate(freq_space):
        b = UnderdampedBrownian(
            __(SCALE * re / dof / (_n + 1), '/cm').au,
            __(SCALE * freq, '/cm').au,
            __(SCALE * width, '/cm').au, distr)
        bosons.append(b)
    for k in range(n_ltc + 1):
        for b in bosons:
            corr.coefficients.append(b.coefficients[k])
            corr.conj_coefficents.append(b.conj_coefficents[k])
            corr.derivatives.append(b.derivatives[k])
    print(f'Inv. Temp.: {distr.beta:.4f}')
    print(corr)

    # HEOM settings:
    dims = [dim for _ in range(corr.k_max)]
    heom_op = Hierachy(h, op, corr, dims)
    s = TensorTrainEDT(rdo, dims, rank=rank)
    # s = TensorTreeEDT(rdo, dims, n_ary=2, rank=rank)

    # Propagator settings:
    callback_steps = 1
    steps = 1000 * callback_steps
    interval = __(0.1 / SCALE / callback_steps, 'fs')
    ps_method = None
    ode_method = 'dopri5'

    # reg_method = 'proper_qr'
    reg_method = 'proper'

    propagator = Propagator(heom_op, s, interval.au, ode_method=ode_method, ps_method=ps_method, reg_method=reg_method)

    fname = (f'brownian-fm{freq_max}-re{re}-w{width}-{dof}x{n_ltc+1}({dim})[{rank}]' + f'-{backend.device}.log')
    meta_string = (f'-{type(s).__name__}-{distr.decomposition_method}-ps{ps_method}-{reg_method}-{ode_method}' +
                   f'-[{log10(backend.ODE_RTOL):+.0f}({log10(backend.ODE_ATOL):+.0f}){log10(backend.SVD_ATOL):+.0f}])')

    if dry_run:
        print('Smoke testing...')
        propagator.step()
        return
    else:
        print(f'Write in `{fname}`.')
        logger1 = Logger(filename=fname, level='info').logger
        logger1.info(f'# {meta_string}')
        logger1.info('# time rdo00 rdo01 rdo10 rdo11')
        logger2 = Logger(filename='_' + fname, level='info').logger
        logger2.info(f'# {meta_string}')
        logger2.info('# time ODE_steps')
        cpu_time0 = time()
        for _n, _t in zip(range(steps), propagator):
            rdo = s.get_rdo()
            t = _t * SCALE
            logger1.info(f'{t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')
            logger2.info(f'{t} {propagator.ode_step_counter[-1]}')

            if _n % callback_steps == 0:
                trace = rdo[0, 0] + rdo[1, 1]
                coh = abs(rdo[0, 1])
                _steps = sum(propagator.ode_step_counter)
                cpu_time = time()
                print(
                    f'({cpu_time - cpu_time0}) {__(t).convert_to("fs").value:.1f} fs | Tr:1{(trace.real - 1):+e}{trace.imag:+e}j | Coh:{coh:f} | ODE steps:{_steps}'
                )
                cpu_time0 = cpu_time
                propagator.ode_step_counter = []
                if coh > 0.55:
                    break

        return


if __name__ == '__main__':
    import os
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Brownian HEOM.')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--freq_max', type=float, default=2000.)
    parser.add_argument('--re', type=float, default=1000.)
    parser.add_argument('--width', type=float, default=50.0)
    parser.add_argument('--dof', type=int, default=4)
    parser.add_argument('--n_ltc', type=int, default=3)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--rank', type=int, default=20)

    # br=50, dof=4, n_ltc: int = 1, dim: int = 20, rank: int = 20

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir))
    args = parser.parse_args()
    print(args)
    test_hierachy(args.freq_max, args.re, args.width, args.dof, args.n_ltc, args.dim, args.rank, args.dry_run)
