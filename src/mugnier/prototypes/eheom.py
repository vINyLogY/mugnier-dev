# coding: utf-8
#include"armadillo.h"
#include<armadillo.h>

from time import time
from typing import Generator, Optional

from mugnier.heom.bath import BoseEinstein, Correlation, Drude, SpectralDensity, UnderdampedBrownian
from mugnier.heom.hierachy import HeomOp, NaiveHierachy, TrainHierachy, TreeHierachy
from mugnier.libs import backend
from mugnier.libs.backend import Array, OptArray
from mugnier.libs.logging import Logger
from mugnier.libs.quantity import Quantity as __
from mugnier.operator.spo import Propagator

inversed_temperature_unit = '/K'
time_unit = 'fs'
energy_unit = '/cm'


def run_spin_boson(
    out: str,
    dry_run: bool,
    # System
    init_rdo: Array,
    elec_bias: float,
    elec_coupling: float,
    # Drudian bath
    include_drude: bool,
    re_d: Optional[float],
    width_d: float,
    # Brownian bath
    include_brownian: bool,
    dof: int,
    freq_max: float,
    re_b: float,
    width_b: float,
    # LTC bath
    temperature: float,
    decomposition_method: str,
    n_ltc: int,
    # Tensor Hierachy Tucker Decompositon
    dim: int,
    htd_method: str,
    rank: int,
    decimation_rate: int,
    # HEOM type
    heom_factor: float,
    ps_method: str,
    ode_method: str,
    reg_method: str,
    use_dvr: bool,
    dvr_space: Optional[tuple[float, float]],
    ht_dim: Optional[int],
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
    p0 = 0.5
    scaling = 1.0 / __(max(abs(elec_bias), abs(elec_coupling)), energy_unit).au
    e = __(elec_bias, energy_unit).au * scaling
    v = __(elec_coupling, energy_unit).au * scaling
    h = backend.array([[-p0 * e, v], [v, (1.0 - p0) * e]])
    op = backend.array([[-p0, 0.0], [0.0, (1.0 - p0)]])

    # Bath settings:
    distr = BoseEinstein(
        n=n_ltc,
        beta=__(1 / temperature, inversed_temperature_unit).au / scaling)
    distr.decomposition_method = decomposition_method
    sds = []  # type:list[SpectralDensity]
    spaces_list = []
    if include_drude:
        drude = Drude(
            __(re_d, energy_unit).au * scaling,
            __(width_d, energy_unit).au * scaling)
        sds.append(drude)
        spaces_list.append(dvr_space)
    if include_brownian:
        freq_space = [freq_max / (dof + 1) * (n + 1) for n in range(dof)]
        for _n, freq in enumerate(freq_space):
            b = UnderdampedBrownian(
                __(re_b / dof / (_n + 1), energy_unit).au * scaling,
                __(freq, energy_unit).au * scaling,
                __(width_b, energy_unit).au * scaling,
            )
            sds.append(b)
            spaces_list.extend([dvr_space, dvr_space])
    spaces = dict(enumerate(spaces_list))
    corr = Correlation(sds, distr)
    corr.fix(roundoff=roundoff)
    print(corr)

    # HEOM settings:
    dims = [ht_dim if k in spaces else dim for k in range(corr.k_max)]
    print(dims)

    if not use_dvr:
        spaces = None
    if htd_method == 'Train':
        s = TrainHierachy(init_rdo,
                          dims,
                          rank=rank,
                          decimation_rate=decimation_rate,
                          spaces=spaces)
    elif htd_method == 'RevTrain':
        s = TrainHierachy(init_rdo,
                          dims,
                          rank=rank,
                          decimation_rate=decimation_rate,
                          rev=False,
                          spaces=spaces)
    elif htd_method == 'Tree2':
        s = TreeHierachy(init_rdo,
                         dims,
                         n_ary=2,
                         rank=rank,
                         decimation_rate=decimation_rate,
                         spaces=spaces)
    elif htd_method == 'Tree3':
        s = TreeHierachy(init_rdo,
                         dims,
                         n_ary=3,
                         rank=rank,
                         decimation_rate=decimation_rate,
                         spaces=spaces)
    elif htd_method == 'Naive':
        s = NaiveHierachy(init_rdo, dims)
    else:
        raise NotImplementedError(f'No htd_method {htd_method}.')
    HeomOp.scaling_factor = heom_factor
    heom_op = HeomOp(s, h, op, corr, dims)

    # Propagator settings:
    steps = int(end / dt) * callback_steps
    interval = __(dt / callback_steps, time_unit) / scaling
    propagator = Propagator(heom_op,
                            s,
                            interval.au,
                            ode_method=ode_method,
                            ps_method=ps_method,
                            reg_method=reg_method)
    if dry_run:
        propagator.step()
    else:
        logger = Logger(filename=out, level='info').logger
        logger.info('# time rdo00 rdo01 rdo10 rdo11')
        cpu_time0 = time()
        for _n, _t in zip(range(steps), propagator):
            if (_n + 1) % callback_steps == 0:
                rdo = s.get_rdo()
                trace = rdo[0, 0] + rdo[1, 1]
                s.opt_update(s.root, s[s.root] / trace)
                rdo /= trace
                t = _t * scaling
                logger.info(
                    f'{t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')

                print(
                    {
                        n: propagator.state.shape(n)
                        for n in propagator._node_visitor
                    },
                    flush=True)
                coh = abs(rdo[0, 1])
                _steps = sum(propagator.ode_step_counter)
                cpu_time = time()
                info = f'[{cpu_time - cpu_time0:.3f} s] {__(t).convert_to("fs").value:.3f} fs | ODE steps:{_steps}'
                info += f' | Tr:1{(trace.real - 1):+e}{trace.imag:+e}j | Coh:{coh:f} | Pop:{abs(rdo[0, 0])}'
                print(info, flush=True)
                cpu_time0 = cpu_time
                propagator.ode_step_counter = []
    return


def run_spin_boson_mix(
    out: str,
    # System
    init_rdo: Array,
    elec_bias: float,
    elec_coupling: float,
    # Drudian bath
    include_drude: bool,
    re_d: Optional[float],
    width_d: float,
    # Brownian bath
    include_brownian: bool,
    dof: int,
    freq_max: float,
    re_b: float,
    width_b: float,
    # LTC bath
    temperature: float,
    decomposition_method: str,
    n_ltc: int,
    # Tensor Hierachy Tucker Decompositon
    dim: int,
    htd_method: str,
    rank: int,
    decimation_rate: int,
    # HEOM type
    heom_factor: float,
    ode_method: str,
    reg_method: str,
    use_dvr: bool,
    dvr_space: Optional[tuple[float, float]],
    ht_dim: Optional[int],
    # Error
    roundoff: float,
    ode_rtol: float,
    ode_atol: float,
    svd_atol: float,
    # Propagator
    dt: tuple[float, float],
    end: tuple[float, float],
    callback_steps: tuple[int, float],
) -> Generator[tuple[float, OptArray], None, None]:

    backend.parameters.ode_rtol = ode_rtol
    backend.parameters.ode_atol = ode_atol
    backend.parameters.svd_atol = svd_atol

    # System settings:
    p0 = 0.5
    scaling = 1.0 / __(elec_bias, energy_unit).au
    e = __(elec_bias, energy_unit).au * scaling
    v = __(elec_coupling, energy_unit).au * scaling
    h = backend.array([[-p0 * e, v], [v, (1.0 - p0) * e]])
    op = backend.array([[-p0, 0.0], [0.0, (1.0 - p0)]])

    # Bath settings:
    distr = BoseEinstein(
        n=n_ltc,
        beta=__(1 / temperature, inversed_temperature_unit).au / scaling)
    distr.decomposition_method = decomposition_method
    sds = []  # type:list[SpectralDensity]
    spaces_list = []
    if include_drude:
        drude = Drude(
            __(re_d, energy_unit).au * scaling,
            __(width_d, energy_unit).au * scaling)
        sds.append(drude)
        spaces_list.append(dvr_space)
    if include_brownian:
        freq_space = [freq_max / (dof + 1) * (n + 1) for n in range(dof)]
        for _n, freq in enumerate(freq_space):
            b = UnderdampedBrownian(
                __(re_b / dof / (_n + 1), energy_unit).au * scaling,
                __(freq, energy_unit).au * scaling,
                __(width_b, energy_unit).au * scaling,
            )
            sds.append(b)
            spaces_list.extend([dvr_space, dvr_space])
    spaces = dict(enumerate(spaces_list))
    corr = Correlation(sds, distr)
    corr.fix(roundoff=roundoff)

    # HEOM settings:
    dims = [ht_dim if k in spaces else dim for k in range(corr.k_max)]
    print(dims)

    if not use_dvr:
        spaces = None
    if htd_method == 'Train':
        s = TrainHierachy(init_rdo,
                          dims,
                          rank=rank,
                          decimation_rate=decimation_rate,
                          spaces=spaces)
    elif htd_method == 'RevTrain':
        s = TrainHierachy(init_rdo,
                          dims,
                          rank=rank,
                          decimation_rate=decimation_rate,
                          rev=False,
                          spaces=spaces)
    elif htd_method == 'Tree2':
        s = TreeHierachy(init_rdo,
                         dims,
                         n_ary=2,
                         rank=rank,
                         decimation_rate=decimation_rate,
                         spaces=spaces)
    elif htd_method == 'Tree3':
        s = TreeHierachy(init_rdo,
                         dims,
                         n_ary=3,
                         rank=rank,
                         decimation_rate=decimation_rate,
                         spaces=spaces)
    elif htd_method == 'Naive':
        s = NaiveHierachy(init_rdo, dims)
    else:
        raise NotImplementedError(f'No htd_method {htd_method}.')
    HeomOp.scaling_factor = heom_factor
    heom_op = HeomOp(s, h, op, corr, dims)

    # Propagator settings:
    logger = Logger(filename=out, level='info').logger
    logger.info('# time rdo00 rdo01 rdo10 rdo11')
    cpu_time0 = time()
    for _ps_method, _dt, _end, _callback in zip(['vmf', 'ps1'], dt, end,
                                                callback_steps):
        _steps = int(_end / _dt) * _callback
        _interval = __(_dt / _callback, time_unit) / scaling
        _propagator = Propagator(heom_op,
                                 s,
                                 _interval.au,
                                 ode_method=ode_method,
                                 ps_method=_ps_method,
                                 reg_method=reg_method)

        for _n, _t in zip(range(_steps), _propagator):
            if (_n + 1) % callback_steps == 0:
                rdo = s.get_rdo()
                trace = rdo[0, 0] + rdo[1, 1]
                #s.opt_update(s.root, s[s.root] / trace)
                #rdo /= trace
                t = _t * scaling
                logger.info(
                    f'{t} {rdo[0, 0]} {rdo[0, 1]} {rdo[1, 0]} {rdo[1, 1]}')

                print(
                    {
                        n: _propagator.state.shape(n)
                        for n in _propagator._node_visitor
                    },
                    flush=True)
                coh = abs(rdo[0, 1])
                _steps = sum(_propagator.ode_step_counter)
                cpu_time = time()
                info = f'[{cpu_time - cpu_time0:.3f} s] {__(t).convert_to("fs").value:.3f} fs | ODE steps:{_steps}'
                info += f' | Tr:1{(trace.real - 1):+e}{trace.imag:+e}j | Coh:{coh:f}'
                print(info, flush=True)
                cpu_time0 = cpu_time
                _propagator.ode_step_counter = []
    return
