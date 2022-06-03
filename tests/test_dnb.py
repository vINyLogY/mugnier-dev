# coding: utf-8

import argparse
import os
from time import time

from mugnier.heom.bath import BoseEinstein, Correlation, Drude, SpectralDensity, UnderdampedBrownian
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT, TensorTreeEDT
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.operator.spo import Propagator


def test_hierachy(out: str, elec_bias: float, elec_coupling: float, freq_max: float, re_b: float, width_b: float,
                  re_d: float, width_d: float, dof: int, n_ltc: int, dim: int, rank: int, decomposition_method: str,
                  htd_method: str, heom_factor: float, ode_rtol: float, ode_atol: float, svd_atol: float,
                  ps_method: str, ode_method: str, reg_method: str, dt: float, end: float, callback_steps: int,
                  dry_run: bool, temperature: float):

    backend.tol.ode_rtol = ode_rtol
    backend.tol.ode_atol = ode_atol
    backend.tol.svd_atol = svd_atol
    print(f'\nBackend settings: {backend.tol} | {backend.device}')

    # System settings:
    scaling = 1.0 / __(elec_bias, '/cm').au
    e = __(elec_bias, '/cm').au * scaling
    v = __(elec_coupling, '/cm').au * scaling
    pop = 0.5
    h = backend.array([[-pop * e, v], [v, (1.0 - pop) * e]])
    op = backend.array([[-pop, 0.0], [0.0, 1.0 - pop]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=n_ltc, beta=__(1 / temperature, '/K').au / scaling)
    distr.decomposition_method = decomposition_method
    drude = Drude(__(re_d, '/cm').au / scaling, __(width_d, '/cm').au / scaling, distr.function)
    sds = [drude]  # type:list[SpectralDensity]
    freq_space = [freq_max / (dof + 1) * (n + 1) for n in range(dof)]
    for _n, freq in enumerate(freq_space):
        b = UnderdampedBrownian(
            __(re_b / dof / (_n + 1), '/cm').au * scaling,
            __(freq, '/cm').au * scaling,
            __(width_b, '/cm').au * scaling,
        )
        sds.append(b)
    corr = Correlation(sds, distr)
    corr.fix()
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
    elif htd_method == 'Naive':
        s = ExtendedDensityTensor(rdo, dims)
    else:
        raise NotImplementedError(f'No htd_method {htd_method}.')

    # Propagator settings:
    steps = int(end / dt) * callback_steps
    interval = __(dt / callback_steps, 'fs') / scaling
    propagator = Propagator(heom_op, s, interval.au, ode_method=ode_method, ps_method=ps_method, reg_method=reg_method)
    if dry_run:
        print('\nSmoke testing...')
        propagator.step()
        return
    else:
        if not out.endswith('.log'):
            out += '.log'
        print(f'\nWrite in `{out}`...')
        logger = Logger(filename=out, level='info').logger
        logger.info('# time rdo00 rdo01 rdo10 rdo11')
        cpu_time0 = time()
        for _n, _t in zip(range(steps), propagator):
            rdo = s.get_rdo()
            t = _t * scaling
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
    f_name = os.path.splitext(os.path.basename(__file__))[0]
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser(description='Drude + Brownian HEOM.')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--re_d', type=float, default=200.0)
    parser.add_argument('--width_d', type=float, default=100.0)
    parser.add_argument('--freq_max', type=float, default=2000.0)
    parser.add_argument('--re_b', type=float, default=1000.0)
    parser.add_argument('--width_b', type=float, default=50.0)
    parser.add_argument('--dof', type=int, default=4)
    parser.add_argument('--n_ltc', type=int, default=3)
    parser.add_argument('--heom_factor', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--decomposition_method', type=str, default='Pade')
    parser.add_argument('--htd_method', type=str, default='Tree2')
    parser.add_argument('--ps_method', type=str, default='vmf')
    parser.add_argument('--ode_method', type=str, default='dopri5')
    parser.add_argument('--reg_method', type=str, default='proper')
    parser.add_argument('--ode_rtol', type=float, default=1.0e-5)
    parser.add_argument('--ode_atol', type=float, default=1.0e-7)
    parser.add_argument('--svd_atol', type=float, default=1.0e-7)
    parser.add_argument('--elec_coupling', type=float, default=0.0)
    parser.add_argument('--elec_bias', type=float, default=5000.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--end', type=float, default=100.0)
    parser.add_argument('--callback_steps', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=300.0)
    parser.add_argument('--out', type=str, default=f_name + '.log')

    args = parser.parse_args()
    kwargs = {arg: getattr(args, arg) for arg in vars(args)}
    for k, v in sorted(kwargs.items()):
        print(f"{k}: {v}")
    test_hierachy(**kwargs)
