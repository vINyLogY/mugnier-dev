# coding: utf-8
"""HEOM with a discrete harmonic oscilato
"""

from time import time
from typing import Callable, Generator, Optional
from matplotlib import pyplot as plt

import numpy as np
import torch
from scipy.special import erf

from mugnier.heom.bath import BoseEinstein, Correlation, DiscreteVibration, Drude, SpectralDensity, UnderdampedBrownian
from mugnier.heom.hierachy import HeomOp, NaiveHierachy, TrainHierachy, TreeHierachy
from mugnier.libs import backend
from mugnier.libs.backend import Array, OptArray
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.operator.spo import Propagator
from mugnier.basis import dvr


def run_dho(
    # System
    ## Elec
    init_wfn: Array,
    ## Nuc
    nuc_dim: int,
    freq: float,
    sys_coupling: float,
    sb_coupling: float,
    # Drudian bath
    include_drude: bool,
    re_d: Optional[float],
    width_d: float,
    # LTC bath
    temperature: float,
    decomposition_method: str,
    n_ltc: int,
    # Tensor Hierachy Tucker Decompositon
    dim: int,
    htd_method: str,
    # HEOM type
    heom_factor: float,
    ode_method: str,
    ps_method: str,
    reg_method: str,
    # Error
    roundoff: float,
    ode_rtol: float,
    ode_atol: float,
    svd_atol: float,
    # Propagator
    dt: float,
    end: float,
    callback_steps: int,
) -> Generator[tuple[float, OptArray], None, None]:

    backend.parameters.ode_rtol = ode_rtol
    backend.parameters.ode_atol = ode_atol
    backend.parameters.svd_atol = svd_atol

    # System settings:
    sigma_z = np.array([[-0.5, 0.0], [0.0, 0.5]])
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])

    # Elec-Nuc
    init_rdo = np.outer(np.conj(init_wfn), init_wfn)
    h = sigma_z + sys_coupling * sigma_x
    op = sigma_z

    # Bath settings:
    distr = BoseEinstein(n=n_ltc, beta=1 / temperature)
    distr.decomposition_method = decomposition_method
    sds = []  # type:list[SpectralDensity]
    if include_drude:
        drude = Drude(re_d, width_d)
        sds.append(drude)
    corr1 = Correlation(sds, distr)
    corr1.fix(roundoff=roundoff)

    corr2 = DiscreteVibration(freq, sb_coupling, beta=None)
    corr = corr1 + corr2
    print(corr)

    # HEOM settings:
    dims = [dim] * corr1.k_max + [nuc_dim] * corr2.k_max
    if htd_method == 'Naive':
        s = NaiveHierachy(init_rdo, dims)
    else:
        raise NotImplementedError(f'No htd_method {htd_method}.')
    HeomOp.scaling_factor = heom_factor
    heom_op = HeomOp(s, h, op, corr, dims)

    # Propagator settings:
    steps = int(end / dt) * callback_steps
    interval = dt / callback_steps
    propagator = Propagator(heom_op,
                            s,
                            interval,
                            ode_method=ode_method,
                            ps_method=ps_method,
                            reg_method=reg_method)

    for _n, _t in zip(range(steps), propagator):
        if (_n + 1) % callback_steps == 0:
            rdo = s.get_rdo()
            trace = torch.trace(rdo)
            s.opt_update(s.root, s[s.root] / trace)
            rdo = s.get_rdo()
            yield _t, rdo

    return


it = run_dho(
    # System
    ## Elec
    init_wfn=[1, 0],
    ## Nuc
    nuc_dim=50,
    freq=0.5,
    sys_coupling=0.2,
    sb_coupling=0.1,
    # Drudian bath
    include_drude=True,
    # include_drude=False,
    re_d=0.5,
    width_d=0.5,
    # LTC bath
    temperature=10,
    decomposition_method='Pade',
    n_ltc=1,
    # Tensor Hierachy Tucker Decompositon
    dim=5,
    htd_method='Naive',
    # HEOM type
    heom_factor=1.0,
    ode_method='dopri5',
    ps_method='vmf',
    reg_method='proper',
    # Error
    roundoff=1.e-8,
    ode_rtol=1.e-5,
    ode_atol=1.e-7,
    svd_atol=1.e-7,
    # Propagator
    dt=0.1,
    end=300,
    callback_steps=1,
)

logger = Logger(filename='4_dho_pure_heom.log', level='info').logger
for _t, rho in it:
    pr = torch.trace(rho @ rho).cpu().numpy()
    print(_t, (pr), flush=True)

    logger.info(f'{_t} {rho[0, 0]} {rho[0, 1]} {rho[1, 0]} {rho[1, 1]}')
