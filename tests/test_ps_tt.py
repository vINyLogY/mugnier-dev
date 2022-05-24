# coding: utf-8

from math import prod
from tqdm import trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT
from mugnier.heom.bath import BoseEinstein, Drude
from mugnier.operator.spo import MasterEqn, Propagator
from mugnier.state.frame import End


def test_hierachy(n: int = 1, dim: int = 20):
    # System settings:
    e = __(5000.0, '/cm').au
    v = __(0.0, '/cm').au
    h = backend.array([[0, v], [v, e]])
    op = backend.array([[0.0, 0.0], [0.0, 1.0]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=n, beta=__(1 / 300, '/K').au)
    distr.decomposition_method = 'Matsubara'
    corr = Drude(__(500, '/cm').au, __(50, '/cm').au, distr)

    # HEOM settings:
    rank = 2 * dim if n != 0 else dim
    dims = [dim] * corr.k_max
    heom_op = Hierachy(h, op, corr, dims)
    s = TensorTrainEDT(rdo, dims, rank=rank)

    # Propagator settings:
    steps = 1000
    interval = __(0.1, 'fs')
    ps_method = None
    ode_method = 'dopri5'
    callback_steps = 1
    reg_method = 'fast'

    propagator = Propagator(
        heom_op, s, interval.au, ode_method=ode_method, ps_method=ps_method, reg_method=reg_method
    )

    fname = f'tt[debug3]-reg_{reg_method}-ps_{ps_method}-{corr.k_max}({dim})[{rank}]-{interval.value}fs-{ode_method}-{backend.device}.log'
    print(f'Write in `{fname}`:')
    logger1 = Logger(filename=fname, level='info').logger
    logger1.info(f'# ODE: {backend.ODE_RTOL}(+{backend.ODE_ATOL}) | PINV:{backend.PINV_TOL}')
    logger1.info('# time_(fs) rdo00 rdo01 rdo10 rdo11')
    it = trange(steps)
    for n, _t in zip(it, propagator):
        rdo = s.get_rdo()
        trace = rdo[0, 0] + rdo[1, 1]
        logger1.info(f'{_t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')

        if n % callback_steps == 0:
            it.set_description(f'Tr:{trace.real:.4e}{trace.imag:+.4e}j | Coh:{abs(rdo[0, 1]):.8f}')


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    for n in [3]:
        for dim in [10]:
            test_hierachy(n=n, dim = dim)
