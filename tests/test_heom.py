# coding: utf-8

from mugnier.heom.bath import (BoseEinstein, Brownian, Correlation, DiscreteVibration, Drude)
from mugnier.heom.hierachy import (ExtendedDensityTensor, Hierachy, SineExtendedDensityTensor, SineHierachy)
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.operator.spo import Integrator
from tqdm import trange

betas = {'HT': __(1 / 100_000, '/K').au, 'ZT': None}

# System settings:
e = __(5000.0, '/cm').au
v = __(0.0, '/cm').au
h = backend.array([[0, v], [v, e]])
op = backend.array([[0.0, 0.0], [0.0, 1.0]])
rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

# Bath settings:
include_drude = False
include_brownian = True

distr = BoseEinstein(n=0, beta=betas['HT'])
corr = Correlation(distr)
if include_drude:
    corr += Drude(__(200, '/cm').au, __(50, '/cm').au, distr)
if include_brownian:
    corr += Brownian(__(500, '/cm').au, __(1500, '/cm').au, __(50, '/cm').au, distr)

# HEOM basis settings
use_dvr = False
scaling_factor = None
dim = 400
dvr_length = 100

# Propagator settings:
callback_steps = 10
steps = 500 * callback_steps
interval = __(0.1 / callback_steps, 'fs')
ode_method = 'dopri5'

# logging
if include_drude and include_brownian:
    prefix = 'all'
elif include_brownian:
    prefix = 'brownian'
elif include_drude:
    prefix = 'drude'
else:
    prefix = 'dummy'
fname = prefix + f'-{"dvr" if use_dvr else "heom"}-{corr.k_max}({dim})'
if use_dvr:
    fname += f'-L{dvr_length}'


def test_hierachy():
    corr.print()
    logger = Logger(filename=fname + '.log', level='info').logger
    logger.info(f'# device:{backend.device} | use_DVR:{use_dvr} | Brownian:{include_brownian} | Drude:{include_drude}')
    logger.info(f'# dim:{dim} | K:{corr.k_max} | ODE_method:{ode_method} | dt:{interval} | callback:{callback_steps}')

    if not use_dvr:
        dims = [dim] * corr.k_max
        Hierachy.scaling_factor = scaling_factor
        heom_op = Hierachy(h, op, corr, dims)
        s = ExtendedDensityTensor(rdo, dims)
    else:
        bases = [(-dvr_length, dvr_length, dim)] * corr.k_max
        SineHierachy.scaling_factor = scaling_factor
        heom_op = SineHierachy(h, op, corr, bases)
        s = SineExtendedDensityTensor(rdo, bases)
    solver = Integrator(heom_op, s)

    logger.info('# time rdo00 rdo01 rdo10 rdo11')
    it = trange(steps)
    for i in it:
        solver.odeint(interval.au, method=ode_method)
        if i % callback_steps == 0:
            _t = interval * i
            _rdo = s.get_rdo()
            logger.info(f'{_t.au} {_rdo[0, 0]} {_rdo[0, 1]} {_rdo[1, 0]} {_rdo[1, 1]}')

            trace = abs(_rdo[0, 0] + _rdo[1, 1])
            coh = abs(_rdo[0, 1])
            it.set_description(f'Time:{_t.value:.2f}fs | Tr:{trace:.4f} | Coh:{coh:.4f}')


if __name__ == '__main__':
    # f_dir = os.path.abspath(os.path.dirname(__file__))
    # os.chdir(f_dir)

    test_hierachy()
