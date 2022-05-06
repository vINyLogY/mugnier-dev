# coding: utf-8

import numpy as np
from tqdm import tqdm, trange
from mugnier.libs import backend
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy
from mugnier.heom.bath import BoseEinstein, Drude
from mugnier.operator.spo import MasterEqn


def test_ps():
    # System settings:
    e = __(5000.0, '/cm').au
    v = __(0.0, '/cm').au
    h = backend.array([[0, v], [v, e]])
    op = backend.array([[0.0, 0.0], [0.0, 1.0]])
    rdo = backend.array([[0.5, 0.5], [0.5, 0.5]])

    # Bath settings:
    distr = BoseEinstein(n=3, beta=__(1 / 300, '/K').au)
    corr = Drude(__(200, '/cm').au, __(50, '/cm').au, distr)

    # HEOM settings:
    dim = 40
    dims = [dim] * corr.k_max
    heom_op = Hierachy(h, op, corr, dims)
    s = ExtendedDensityTensor(rdo, dims)
    solver = MasterEqn(heom_op, s)

    func = solver.node_eom(s.root)
    diff_norm = np.linalg.norm(func(s[s.root]).cpu().numpy())
    print(diff_norm)
    np.testing.assert_almost_equal(0.016109077, diff_norm)


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    test_ps()
