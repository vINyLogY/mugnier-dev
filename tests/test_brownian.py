# coding: utf-8

import enum
from math import log, log10, prod
import sys
from tqdm import trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT, TensorTreeEDT
from mugnier.heom.bath import BoseEinstein, Correlation, Drude, UnderdampedBrownian
from mugnier.operator.spo import MasterEqn, Propagator
from mugnier.state.frame import End

SCALE = 1 / __(5000.0, '/cm').au
print('Elec. bias: ', SCALE)


def test_hierachy(dof=4, n_ltc: int = 1, dim: int = 20, rank: int = 20):
    # System settings:
    e = __(5000.0 * SCALE, '/cm').au
    v = __(0.0 * SCALE, '/cm').au
    pop = 0.5
    h = backend.array([[-pop * e, v], [v, (1.0 - pop) * e]])
    op = backend.array([[-pop, 0.0], [0.0, 1.0 - pop]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])
    print('Scaled Elec. H: ', h)

    # Bath settings:
    distr = BoseEinstein(n=n_ltc, beta=__(1 / SCALE / 300, '/K').au)
    distr.decomposition_method = 'Pade'
    corr = Correlation(distr)

    freq_max = 2000
    freq_space = [freq_max / (dof + 1) * (n + 1) for n in range(dof)]
    bosons = []  # type:list[UnderdampedBrownian]
    for _n, freq in enumerate(freq_space):
        b = UnderdampedBrownian(
            __(SCALE * 1000 / dof / (_n + 1), '/cm').au,
            __(SCALE * freq, '/cm').au,
            __(SCALE * 50, '/cm').au, distr)
        bosons.append(b)
    for k in range(n_ltc + 1):
        for b in bosons:
            corr.coefficients.append(b.coefficients[k])
            corr.conj_coefficents.append(b.conj_coefficents[k])
            corr.derivatives.append(b.derivatives[k])
    corr.print()
    print('Inv. Temp.: ', distr.beta)

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

    fname = (
        f'brownian-{type(s).__name__}-{distr.decomposition_method}-{reg_method}-ps{ps_method}-{dof}x{n_ltc+1}({dim})[{rank}]'
        +
        f'-{ode_method}-[{log10(backend.ODE_RTOL):+.0f}({log10(backend.ODE_ATOL):+.0f}){log10(backend.SVD_ATOL):+.0f}])'
        + f'-{backend.device}.log')
    print(s.shape(s.root))
    print(f'Write in `{fname}`:', file=sys.stderr)
    logger1 = Logger(filename=fname, level='info').logger
    logger1.info('# time rdo00 rdo01 rdo10 rdo11')
    logger2 = Logger(filename='metas_' + fname, level='info').logger
    logger2.info(f'# ODE: {backend.ODE_RTOL}(+{backend.ODE_ATOL}) | PINV:{backend.SVD_ATOL}')
    logger2.info('# time ODE_steps')
    it = trange(steps)
    for _n, _t in zip(it, propagator):
        rdo = s.get_rdo()
        t = _t * SCALE
        trace = rdo[0, 0] + rdo[1, 1]
        logger1.info(f'{t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')
        logger2.info(f'{t} {propagator.ode_step_counter[-1]}')

        if _n % callback_steps == 0:
            coh = abs(rdo[0, 1])
            _steps = sum(propagator.ode_step_counter)
            it.set_description(f'Tr:1{(trace.real - 1):+.4e}{trace.imag:+.4e}j | Coh:{coh:.8f} | ODE steps:{_steps}')
            propagator.ode_step_counter = []
            if coh > 0.55 or _steps > 10000:
                break


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'brownian_data'))

    #for i in [2, 3]:
    dim = 10
    rank = 20
    test_hierachy(dof=4, n_ltc=1, dim=dim, rank=rank)
