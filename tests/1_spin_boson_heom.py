# coding: utf-8

import argparse
import os
from mugnier.libs.backend import array

from mugnier.prototypes import heom

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser(description='Drude + Brownian HEOM.')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')

    parser.add_argument('--elec_bias', type=float, default=5000.0)
    parser.add_argument('--elec_coupling', type=float, default=0.0)

    parser.add_argument('--include_drude', type=bool, default=True)
    parser.add_argument('--re_d', type=float, default=200.0)
    parser.add_argument('--width_d', type=float, default=100.0)

    parser.add_argument('--include_brownian', type=bool, default=False)
    parser.add_argument('--dof', type=int, default=4)
    parser.add_argument('--freq_max', type=float, default=2000.0)
    parser.add_argument('--re_b', type=float, default=1000.0)
    parser.add_argument('--width_b', type=float, default=50.0)

    parser.add_argument('--temperature', type=float, default=100.0)
    parser.add_argument('--decomposition_method', type=str, default='Pade')
    parser.add_argument('--n_ltc', type=int, default=3)

    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--htd_method', type=str, default='Tree2')
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--decimation_rate', type=int, default=10)

    parser.add_argument('--heom_factor', type=float, default=1.0)
    parser.add_argument('--ps_method', type=str, default='vmf')
    parser.add_argument('--ode_method', type=str, default='bosh3')
    parser.add_argument('--reg_method', type=str, default='proper')

    parser.add_argument('--roundoff', type=float, default=1.0e-8)
    parser.add_argument('--ode_rtol', type=float, default=1.0e-5)
    parser.add_argument('--ode_atol', type=float, default=1.0e-7)
    parser.add_argument('--svd_atol', type=float, default=1.0e-7)

    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--end', type=float, default=100.0)
    parser.add_argument('--callback_steps', type=int, default=1)

    args = parser.parse_args()
    kwargs = {arg: getattr(args, arg) for arg in vars(args)}
    for k, v in sorted(kwargs.items()):
        print(f"{k}: {v}")

    out = kwargs.pop('out')
    if out is None:
        out = os.path.splitext(os.path.basename(__file__))[0]
    if not out.endswith('.log'):
        out += '.log'
    print(f'Write in `{out}`...')

    kwargs['init_rdo'] = array([[0.5, 0.5], [0.5, 0.5]])
    kwargs['out'] = out
    heom.run_spin_boson(**kwargs)
