# coding: utf-8

import enum
from math import log, log10, prod
from time import time
from tqdm import trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT, TensorTreeEDT
from mugnier.heom.bath import BoseEinstein, Correlation, Drude, SpectralDensity, UnderdampedBrownian
from mugnier.operator.spo import MasterEqn, Propagator
from mugnier.state.frame import End


def test_hierachy(
    out_filename: str,
    elec_bias: float = 5000.0,
    elec_coupling: float = 0.0,
    re: float = 200.0,
    width: float = 100.0,
    n_ltc: int = 3,
    dim: int = 10,
    rank: int = 20,
    decomposition_method: str = 'Pade',
    htd_method: str = 'Train',
    heom_factor: float = 2,
    ode_rtol: float = 1.0e-5,
    ode_atol: float = 1.0e-8,
    svd_atol: float = 1.0e-8,
    ps_method: str = 'vmf',
    ode_method: str = 'dopri5',
    reg_method: str = 'proper',
    dt: float = 0.1,
    end: float = 100.0,
    callback_steps: int = 1,
    dry_run: bool = False,
):
    print(f'HT({htd_method}) | DC({decomposition_method})' +
          f' | PS({ps_method}) | REG({reg_method}) | ODE({ode_method})' +
          f' | HEOM({heom_factor:.2f}) | {backend.tol}')
    backend.tol.ode_rtol = ode_rtol
    backend.tol.ode_atol = ode_atol
    backend.tol.svd_atol = svd_atol

    # System settings:
    SCALE = 1 / __(elec_bias, '/cm').au
    e = __(elec_bias * SCALE, '/cm').au
    v = __(elec_coupling * SCALE, '/cm').au
    pop = 0.5
    h = backend.array([[-pop * e, v], [v, (1.0 - pop) * e]])
    op = backend.array([[-pop, 0.0], [0.0, 1.0 - pop]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=n_ltc, beta=__(1 / SCALE / 300, '/K').au)
    distr.decomposition_method = decomposition_method
    print(distr)
    drude = Drude(__(SCALE * re, '/cm').au, __(SCALE * width, '/cm').au, distr.function)
    corr = Correlation([drude], distr)
    print(corr)

    # HEOM settings:
    dims = [dim for _ in range(corr.k_max)]
    Hierachy.scaling_factor = heom_factor
    heom_op = Hierachy(h, op, corr, dims)
    if htd_method == 'Train':
        s = TensorTrainEDT(rdo, dims, rank=rank)
    elif htd_method == 'RevTrain':
        s = TensorTrainEDT(rdo, dims, rank=rank, rev=False)
    elif htd_method == 'Tree2':
        s = TensorTreeEDT(rdo, dims, n_ary=2, rank=rank)
    elif htd_method == 'Tree3':
        s = TensorTreeEDT(rdo, dims, n_ary=3, rank=rank)
    else:
        s = ExtendedDensityTensor(rdo, dims)

    # Propagator settings:
    steps = int(end / dt) * callback_steps
    interval = __(0.1 / SCALE / callback_steps, 'fs')
    propagator = Propagator(heom_op, s, interval.au, ode_method=ode_method, ps_method=ps_method, reg_method=reg_method)
    if dry_run:
        print('Smoke testing...')
        propagator.step()
        return
    else:
        if not out_filename.endswith('.log'):
            out_filename += '.log'
        print(f'Write in `{out_filename}`...')
        logger = Logger(filename=out_filename, level='info').logger
        logger.info('# time rdo00 rdo01 rdo10 rdo11')
        cpu_time0 = time()
        for _n, _t in zip(range(steps), propagator):
            rdo = s.get_rdo()
            t = _t * SCALE
            logger.info(f'{t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')
            if _n % callback_steps == 0:
                trace = rdo[0, 0] + rdo[1, 1]
                coh = abs(rdo[0, 1])
                _steps = sum(propagator.ode_step_counter)
                cpu_time = time()
                info = f'[{cpu_time - cpu_time0:.3f} s] {__(t).convert_to("fs").value:.1f} fs | ODE steps:{_steps}'
                info += f' | Tr:1{(trace.real - 1):+e}{trace.imag:+e}j | Coh:{coh:f}'
                print(info)
                cpu_time0 = cpu_time
                propagator.ode_step_counter = []
                if coh > 0.55:
                    break
        return


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Drudian HEOM.')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--re', type=float, default=10000.)
    parser.add_argument('--width', type=float, default=50.0)
    parser.add_argument('--n_ltc', type=int, default=3)
    parser.add_argument('--heom_factor', type=float, default=2.0)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--rank', type=int, default=20)

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir))
    args = parser.parse_args()
    kwargs = {}
    for arg in vars(args):
        kwargs[arg] = getattr(args, arg)
    fname = f"drude-re{args.re}-w{args.width}-{args.n_ltc}({args.dim})[{args.rank}]-{backend.device}.log"
    test_hierachy(fname, **kwargs)
