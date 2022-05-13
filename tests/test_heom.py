# coding: utf-8

from tqdm import trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy, TensorTrainEDT
from mugnier.heom.bath import BoseEinstein, Drude
from mugnier.operator.spo import MasterEqn, Propagator


def test_hierachy():
    # System settings:
    e = __(5000.0, '/cm').au
    v = __(0.0, '/cm').au
    h = backend.array([[0, v], [v, e]])
    op = backend.array([[0.0, 0.0], [0.0, 1.0]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=2, beta=__(1 / 300, '/K').au)
    corr = Drude(__(500, '/cm').au, __(50, '/cm').au, distr)

    # HEOM settings:
    dim = 20
    dims = [dim] * corr.k_max
    heom_op = Hierachy(h, op, corr, dims)
    s = ExtendedDensityTensor(rdo, dims)

    # Propagator settings:
    steps = 100
    interval = __(0.01, 'fs')

    propagator = Propagator(heom_op, s, interval.au)
    logger1 = Logger(filename=f'naive_heom_{corr.k_max}({dim})-{interval.value:.04f}fs-{backend.device}.log',
                     level='info').logger
    logger1.info('# time_(fs) rdo00 rdo01 rdo10 rdo11')
    it = trange(steps)
    for n in it:
        propagator.step()
        _t = n * interval.value
        rdo = s.get_rdo()
        trace = rdo[0, 0] + rdo[1, 1]
        logger1.info(f'{_t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')
        it.set_description(f'Tr:{trace} | Coh:{abs(rdo[0, 1])}')


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    test_hierachy()
