from mugnier.basis.dvr import SineDVR
from mugnier.heom.bath import (BoseEinstein, Correlation, DiscreteVibration, Drude, UnderdampedBrownian)
from mugnier.heom.hierachy import Hierachy, ExtendedDensityTensor
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.operator.spo import MasterEqn, Propagator
from tqdm import trange

betas = {'HT': __(1 / 100_000, '/K').au, 'ZT': None, 'RT': __(1 / 300, '/K').au}

# System settings:
e = __(5000.0, '/cm').au
v = __(0.0, '/cm').au
h = backend.array([[0, v], [v, e]])
op = backend.array([[0.0, 0.0], [0.0, 1.0]])
rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

# Bath settings:
beta = betas['RT']
distr = BoseEinstein(n=3, beta=beta)
corr = DiscreteVibration(__(1500, '/cm').au, __(500, '/cm').au, distr)
dim = 200

# Propagator settings:
steps = 1000
interval = __(0.01, 'fs')
ode_method = 'dopri5'

# logging
fname = 'HEOM' + f'-({dim})-{beta:.4f}'


def iter_lvn():
    logger = Logger(filename=fname + '.log', level='info').logger
    logger.info(f'# device:{backend.device}')
    logger.info(f'# dim:{dim} | K:{corr.k_max} | ODE_method:{ode_method} | dt:{interval} ')

    lvn_op = Hierachy(h, op, corr, [dim] * corr.k_max)
    s = ExtendedDensityTensor(rdo, [dim] * corr.k_max)
    print(s.shape(s.root))

    solver = Propagator(lvn_op, s, interval.au, ode_method=ode_method)

    logger.info('# time rdo00 rdo01 rdo10 rdo11')
    it = trange(steps)
    for i in it:
        solver.direct_step()
        _t = interval * i
        _rdo = s.get_rdo()
        logger.info(f'{_t.au} {_rdo[0, 0]} {_rdo[0, 1]} {_rdo[1, 0]} {_rdo[1, 1]}')

        trace = abs(_rdo[0, 0] + _rdo[1, 1])
        coh = abs(_rdo[0, 1])
        it.set_description(f'Time:{_t.value:.2f}fs | Tr:{trace:.4f} | Coh:{coh:.4f}')


iter_lvn()