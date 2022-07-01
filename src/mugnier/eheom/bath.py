#!/usr/bin/env python
# coding: utf-8
"""
Decomposition of the bath and BE distribution
"""
from __future__ import annotations
from typing import Callable, Literal, Optional

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

    def function(self, w: complex) -> complex:
        beta = self.beta
        if beta is None:
            return 1.0
        else:
            return 1.0 / (1.0 - np.exp(-beta * w))

    @property
    def residues(self) -> list[tuple[complex, complex]]:
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
    def matsubara(n: int) -> tuple[Array, Array]:
        zetas = [2.0 * PI * (i + 1) for i in range(n)]
        residues = [1.0] * n
        return array(residues), array(zetas)

    @staticmethod
    def pade1(n: int) -> tuple[Array, Array]:
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

    def __init__(self, spectral_densities: list[SpectralDensity], distribution: BoseEinstein) -> None:
        self.spectral_densities = spectral_densities
        self.distribution = distribution

        self.sym_coefficients = list()  # type: list[complex]
        self.asym_coefficients = list()  # type: list[complex]
        self.derivatives = list()  # type: list[complex]

        for sd in spectral_densities:
            ss, aa, dd = sd.get_htc(distr=distribution)
            self.sym_coefficients.extend(ss)
            self.asym_coefficients.extend(aa)
            self.derivatives.extend(dd)
        self.get_ltc()
        return

    def fix(self, roundoff) -> None:
        """Fix underflow in the coefficients."""

        def _fix(num: complex) -> complex:
            re = 0.0 if abs(num.real) < roundoff else num.real
            im = 0.0 if abs(num.imag) < roundoff else num.imag
            return re + 1.0j * im

        self.sym_coefficients = list(map(_fix, self.sym_coefficients))
        self.asym_coefficients = list(map(_fix, self.asym_coefficients))
        self.derivatives = list(map(_fix, self.derivatives))
        return

    @property
    def k_max(self) -> int:
        return len(self.derivatives)

    def get_ltc(self) -> None:
        residues = self.distribution.residues
        for res, pole in residues:
            d = -1.0j * pole
            cs = [-2.0j * PI * res * sd.function(pole) for sd in self.spectral_densities if callable(sd.function)]
            if cs:
                c = np.sum(cs)
                self.sym_coefficients.append((c + np.conj(c)) / 2.0)
                self.asym_coefficients.append((c - np.conj(c)) / 2.0)
                self.derivatives.append(d)
        return

    def __str__(self) -> None:
        k = self.k_max
        if k > 0:
            string = f"Correlation {k} * ( s | a | g ):"
            for s, a, g in zip(self.sym_coefficients, self.asym_coefficients, self.derivatives):
                string += f"\n{s.real:+.4e}{s.imag:+.4e}j | {a.real:+.4e}{a.imag:+.4e}j | {g.real:+.4e}{g.imag:+.4e}j"
        else:
            string = 'Empty Correlation object'
        return string


class SpectralDensity:

    def function(self, w: complex) -> complex:
        pass

    def get_htc(self, distr: BoseEinstein) -> tuple[list[complex], list[complex], list[complex]]:
        pass


class Drude(SpectralDensity):

    def __init__(self, reorganization_energy: float, relaxation: float) -> None:
        self.l = reorganization_energy
        self.g = relaxation
        return

    def function(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return (2.0 / PI) * l * g * w / (w**2 + g**2)

    def get_htc(self, distr: BoseEinstein) -> tuple[list[complex], list[complex], list[complex]]:
        _c = -2.0j * self.l * self.g * distr.function(-1.0j * self.g)
        _d = -self.g
        sym_coeff = [_c + np.conj(_c) / 2.0]
        asym_coeff = [_c - np.conj(_c) / 2.0]
        derivatives = [_d]
        return sym_coeff, asym_coeff, derivatives


class DiscreteVibration(SpectralDensity):

    def __init__(self, frequency: float, coupling: float, beta: Optional[float]) -> None:
        self.w0 = frequency
        self.g = coupling
        self.beta = beta
        return

    def get_htc(self, distr: BoseEinstein) -> tuple[list[complex], list[complex], list[complex]]:
        w0 = self.w0
        g = self.g
        beta = distr.beta
        coth = 1.0 / np.tanh(beta * w0 / 2.0) if beta is not None else 1.0
        c1 = g**2 / 2.0 * (coth + 1.0)
        c2 = g**2 / 2.0 * (coth - 1.0)
        sym_coeff = [(c1 + c2) / 2.0, (c2 + c1) / 2.0]
        asym_coeff = [(c1 - c2) / 2.0, (c2 - c1) / 2.0]
        derivatives = [-1.0j * w0, +1.0j * w0]
        return sym_coeff, asym_coeff, derivatives


class UnderdampedBrownian(SpectralDensity):

    def __init__(self, reorganization_energy: float, frequency: float, relaxation: float) -> None:
        self.w0 = frequency
        self.g = relaxation
        self.l = reorganization_energy
        return

    def function(self, w: complex) -> complex:
        l = self.l
        g = self.g
        w0 = self.w0
        return (4.0 / PI) * l * g * (w0**2 + g**2) * w / ((w + w0)**2 + g**2) / ((w - w0)**2 + g**2)

    def get_htc(self, distr: BoseEinstein) -> tuple[list[complex], list[complex], list[complex]]:
        f = distr.function
        l = self.l
        g = self.g
        w0 = self.w0

        a = l * (w0**2 + g**2) / w0
        c1 = +a * f(-1.0j * (g + 1.0j * w0))
        c2 = -a * f(-1.0j * (g - 1.0j * w0))

        sym_coeff = [(c1 + np.conj(c2)) / 2.0, (c2 + np.conj(c1)) / 2.0]
        asym_coeff = [(c1 - np.conj(c2)) / 2.0, (c2 - np.conj(c1)) / 2.0]
        derivatives = [-(g + 1.0j * w0), -(g - 1.0j * w0)]
        return sym_coeff, asym_coeff, derivatives
