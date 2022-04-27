#!/usr/bin/env python
# coding: utf-8
"""
Decomposition of the bath and BE distribution
"""
from __future__ import annotations

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

        self.get_correlation()
        self.get_correction()
        return

    @property
    def k_max(self) -> int:
        return len(self.derivatives)

    def spectral_density(self, w: complex) -> complex:
        raise NotImplementedError

    def get_correlation(self) -> None:
        pass

    def get_correction(self) -> None:
        pass

    def print(self) -> None:
        string = f"""Correlation coefficents:
            c: {array(self.coefficients)};
            (c* = {array(self.conj_coefficents)};)
            gamma: {array(self.derivatives)}.
        """
        print(string)

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

    def correction(self) -> Tuple[list[complex], list[complex], list[complex]]:
        zipped = [
            (-2.0j * PI * res * self.spectral_density(pole), -1.0j * pole) for res, pole in self.distribution.residues
        ]
        _cs, _ds = zip(*zipped)

        self.coefficients.extend(_cs)
        self.conj_coefficents.extend(np.conj(_c) for _c in _cs)
        self.derivatives.extend(_ds)

        return


class DiscreteVibration(Correlation):

    def __init__(self, frequency: float, coupling: float, distribution: BoseEinstein) -> None:
        self.w = frequency
        self.g = coupling

        super().__init__(distribution)
        return

    def spectral_density(self, w: complex) -> complex:
        return 0.0

    def get_correlation(self) -> None:
        beta = self.distribution.beta
        w = self.w
        g = self.g
        f = 1.0 / np.tanh(beta * w / 2.0) if beta is not None else 1.0

        self.coefficients.extend([g**2 / 2.0 * (f + 1.0), g**2 / 2.0 * (f - 1.0)])
        self.conj_coefficents.extend([g**2 / 2.0 * (f - 1.0), g**2 / 2.0 * (f + 1.0)])
        self.derivatives.extend([-1.0j * w, +1.0j * w])
        return
