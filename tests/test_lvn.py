from mugnier.basis.dvr import SineDVR
from mugnier.heom.bath import (BoseEinstein, Correlation, DiscreteVibration, Drude, UnderdampedBrownian)
from mugnier.mctdh.lvn import SpinBosonDensityOperator, SpinBosonLvN
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

distr = BoseEinstein(n=0, beta=None)
corr = DiscreteVibration(__(1500, '/cm').au, __(500, '/cm').au, distr)
dim = 200

# Propagator settings:
callback_steps = 10
steps = 500 * callback_steps
interval = __(0.1 / callback_steps, 'fs')
ode_method = 'dopri5'

# logging
fname = 'LvN' + f'-({dim})'


def iter_lvn():
    logger = Logger(filename=fname + '.log', level='info').logger
    logger.info(f'# device:{backend.device}')
    logger.info(f'# dim:{dim} | K:{corr.k_max} | ODE_method:{ode_method} | dt:{interval} | callback:{callback_steps}')

    lvn_op = SpinBosonLvN(h, op, [corr], [dim])
    s = SpinBosonDensityOperator(rdo, [dim])
    #print(s.shape(s.root))

    solver = Integrator(lvn_op, s)

    logger.info('# time rdo00 rdo01 rdo10 rdo11')
    it = trange(steps)
    for i in it:
        solver.odeint(interval.au, method=ode_method)
        if i % callback_steps == 0:
            _t = interval * i
            _rdo = s.get_rdo()
            logger.info(f'{_t.au} {_rdo[0, 0]} {_rdo[0, 1]} {_rdo[1, 0]} {_rdo[1, 1]}')

            yield s[s.root]

            trace = abs(_rdo[0, 0] + _rdo[1, 1])
            coh = abs(_rdo[0, 1])
            it.set_description(f'Time:{_t.value:.2f}fs | Tr:{trace:.4f} | Coh:{coh:.4f}')


for _ in iter_lvn():
    pass