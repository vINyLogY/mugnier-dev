#!/usr/bin/env python
# coding: utf-8
"""
Decomposition of the bath and BE distribution
"""
from __future__ import annotations
from math import gamma

from typing import Literal, Optional, Tuple

from mugnier.libs.backend import PI, Array, array, np


def _tridiag_eigsh(subdiag: Array) -> Array:
    mat = np.diag(subdiag, -1) + np.diag(subdiag, 1)
    return np.sort(np.linalg.eigvalsh(mat))[::-1]


class BoseEinstein(object):
    decomposition_method = 'Pade'  # type: Literal['Pade', 'Matsubara']
    pade_type = '(N-1)/N'  # type: Literal['(N-1)/N']

    def __init__(self, n: int = 0, beta: Optional[float] = None) -> None:
        """
        Args: 
            n: Number of low-temperature correction terms.
            beta: inversed temperature, `None` indicates zero temperature. 
        """
        if beta is not None:
            assert beta >= 0

        self.n = n
        self.beta = beta

        return

    def __str__(self) -> str:
        if self.decomposition_method == 'Pade':
            info = f'Padé[{self.pade_type}]'
        else:
            info = self.decomposition_method
        return f'Bose-Einstein at ß = {self.beta:.4f} ({info}; N={self.n})'

    def func(self, w: complex) -> complex:
        beta = self.beta
        if beta is None:
            return 1.0
        else:
            return 1.0 / (1.0 - np.exp(-beta * w))

    @property
    def residues(self) -> list[Tuple[complex, complex]]:
        method = NotImplemented
        if self.decomposition_method == 'Pade':
            if self.pade_type == '(N-1)/N':
                method = self.pade1
        elif self.decomposition_method == 'Matsubara':
            method = self.matsubara

        if method is NotImplemented:
            raise NotImplementedError

        n = self.n
        b = self.beta
        if b is None or n == 0:
            ans = []
        else:
            residues, zetas = method(n)
            ans = [(r / b, -1.0j * z / b) for r, z in zip(residues, zetas)]
        return ans

    @staticmethod
    def matsubara(n: int) -> Tuple[Array, Array]:
        zetas = [2.0 * PI * (i + 1) for i in range(n)]
        residues = [1.0] * n
        return array(residues), array(zetas)

    @staticmethod
    def pade1(n: int) -> Tuple[Array, Array]:
        # (N-1)/N method
        assert n > 0

        subdiag_q = array([1.0 / np.sqrt((2 * i + 3) * (2 * i + 5)) for i in range(2 * n - 1)])
        zetas = 2.0 / _tridiag_eigsh(subdiag_q)[:n]
        roots_q = np.power(zetas, 2)

        subdiag_p = array([1.0 / np.sqrt((2 * i + 5) * (2 * i + 7)) for i in range(2 * n - 2)])
        roots_p = np.power(2.0 / _tridiag_eigsh(subdiag_p)[:n - 1], 2)

        residues = np.zeros((n,))
        for i in range(n):
            res_i = 0.5 * n * (2 * n + 3)
            if i < n - 1:
                res_i *= (roots_p[i] - roots_q[i]) / (roots_q[n - 1] - roots_q[i])
            for j in range(n - 1):
                if j != i:
                    res_i *= ((roots_p[j] - roots_q[i]) / (roots_q[j] - roots_q[i]))
            residues[i] = res_i

        return array(residues), array(zetas)


class Correlation(object):

    def __init__(self, distribution: BoseEinstein) -> None:
        self.distribution = distribution

        self.coefficients = list()  # type: list[complex]
        self.conj_coefficents = list()  # type: list[complex]
        self.derivatives = list()  # type: list[complex]
        self.closure = None
        return

    def fix(self) -> None:

        def _fix(num: complex, roundoff: float = 1.0e-8) -> complex:
            re = 0.0 if abs(num.real) < roundoff else num.real
            im = 0.0 if abs(num.imag) < roundoff else num.imag
            return re + 1.0j * im

        self.coefficients = map(_fix, self.coefficients)
        self.conj_coefficents = map(_fix, self.conj_coefficents)
        self.derivatives = map(_fix, self.derivatives)
        return


    @property
    def k_max(self) -> int:
        return len(self.derivatives)

    def spectral_density(self, w: complex) -> complex:
        raise NotImplementedError

    def get_correlation(self) -> None:
        raise NotImplementedError

    def get_correction(self) -> None:
        zipped = [
            (-2.0j * PI * res * self.spectral_density(pole), -1.0j * pole) for res, pole in self.distribution.residues
        ]
        if zipped:
            _cs, _ds = zip(*zipped)
            self.coefficients.extend(_cs)
            self.conj_coefficents.extend(np.conj(_c) for _c in _cs)
            self.derivatives.extend(_ds)

        return

    def __str__(self) -> None:
        string = "Correlation (c | c* | g):\n"
        for c, cc, g in zip(self.coefficients, self.conj_coefficents, self.derivatives):
            string += f"{c.real:+.4e}{c.imag:+.4e}j | {cc.real:+.4e}{cc.imag:+.4e}j | {g.real:+.4e}{g.imag:+.4e}j\n"
        return string

    def __add__(self, other: Correlation) -> Correlation:
        assert isinstance(other, Correlation)
        assert self.distribution.beta == other.distribution.beta
        cls = type(self)
        distr = type(self.distribution)
        obj = cls(distribution=distr(beta=self.distribution.beta))
        for attr in ['coefficients', 'conj_coefficents', 'derivatives']:
            setattr(obj, attr, getattr(self, attr) + getattr(other, attr))
        return obj


class Drude(Correlation):

    def __init__(self, reorganization_energy: float, relaxation: float, distribution: BoseEinstein) -> None:
        self.l = reorganization_energy
        self.g = relaxation

        super().__init__(distribution)
        self.get_correlation()
        self.get_correction()
        return

    def spectral_density(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return (2.0 / PI) * l * g * w / (w**2 + g**2)

    def get_correlation(self) -> None:
        f = self.distribution.func
        _c = -2.0j * self.l * self.g * f(-1.0j * self.g)
        _d = -self.g
        self.coefficients.append(_c)
        self.conj_coefficents.append(np.conj(_c))
        self.derivatives.append(_d)

        return


class DiscreteVibration(Correlation):
    rtol = 0.01

    def __init__(self, frequency: float, coupling: float, distribution: BoseEinstein) -> None:
        self.w0 = frequency
        self.g = coupling

        super().__init__(distribution)
        self.get_correlation()
        return

    def spectral_density(self, w: complex) -> complex:
        w0 = self.w0
        tol = self.rtol * w0
        return self.g**2 / (2.0 * tol) if abs(w - w0) < tol else 0.0

    def get_correlation(self) -> None:
        beta = self.distribution.beta
        w0 = self.w0
        g = self.g
        f = 1.0 / np.tanh(beta * w0 / 2.0) if beta is not None else 1.0

        self.coefficients.extend([g**2 / 2.0 * (f + 1.0), g**2 / 2.0 * (f - 1.0)])
        self.conj_coefficents.extend([g**2 / 2.0 * (f - 1.0), g**2 / 2.0 * (f + 1.0)])
        self.derivatives.extend([-1.0j * w0, +1.0j * w0])
        return

    def get_correction(self) -> None:
        """No corrections for discrete vibrations"""
        return


class UnderdampedBrownian(Correlation):

    def __init__(self, reorganization_energy: float, frequency: float, relaxation: float,
                 distribution: BoseEinstein) -> None:
        self.w0 = frequency
        self.g = relaxation
        self.l = reorganization_energy

        super().__init__(distribution)
        self.get_correlation()
        self.get_correction()
        return

    def spectral_density(self, w: complex) -> complex:
        l = self.l
        g = self.g
        w0 = self.w0
        return (4.0 / PI) * l * g * (w0**2 + g**2) * w / ((w + w0)**2 + g**2) / ((w - w0)**2 + g**2)

    def get_correlation(self) -> None:
        f = self.distribution.func
        l = self.l
        g = self.g
        w0 = self.w0

        a = l * (w0**2 + g**2) / w0
        c1 = +a * f(-1.0j * (g + 1.0j * w0))
        c2 = -a * f(-1.0j * (g - 1.0j * w0))

        self.coefficients.extend([c1, c2])
        self.conj_coefficents.extend([np.conj(c2), np.conj(c1)])
        self.derivatives.extend([-(g + 1.0j * w0), -(g - 1.0j * w0)])
        return
