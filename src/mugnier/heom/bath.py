#!/usr/bin/env python
# coding: utf-8
"""
Decomposition of the bath and BE distribution
"""
from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

from mugnier.libs.backend import Array, array, PI, np


def tridiag_eigsh(subdiag: Array) -> Array:
    mat = np.diag(subdiag, -1) + np.diag(subdiag, 1)
    return np.sort(np.linalg.eigvalsh(mat))[::-1]


class Correlation(object):

    def __init__(self, spd: SpectrualDensity, distribution: BoseEinstein):
        """
        n : int
            number of the terms in the basis functions
        beta : float, optional
            Inverse temperature; 1 / (k_B T); `None` indicates zero temperature.
        """
        self.spd = spd
        self.distribution = distribution

        cs, ccs, ds = spd.hta_correlations(distribution.func)

        for res, pole in distribution.residues:
            _c = -2.0j * PI * res * spd.func(pole)
            _d = -1.0j * pole
            cs.append(_c)
            ccs.append(np.conj(_c))
            ds.append(_d)

        self.coefficients = array(cs)
        self.conj_coefficents = array(ccs)
        self.derivatives = array(ds)
        return


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
        zetas = 2.0 / tridiag_eigsh(subdiag_q)[:n]
        roots_q = np.power(zetas, 2)

        subdiag_p = array([1.0 / np.sqrt((2 * i + 5) * (2 * i + 7)) for i in range(2 * n - 2)])
        roots_p = np.power(2.0 / tridiag_eigsh(subdiag_p)[:n - 1], 2)

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


class SpectrualDensity(object):

    def __init__(self):
        pass

    def func(self, w: complex) -> complex:
        return NotImplemented

    def hta_correlations(
        self,
        distribution_func: Callable[[complex], complex],
    ) -> Tuple[list[complex], list[complex], list[complex]]:
        """
        Returns: 
            coefficients, conjugate coefficients, derivatives
        """
        raise NotImplementedError


class Drude(SpectrualDensity):

    def __init__(self, reorganization_energy, relaxation) -> None:
        self.l = reorganization_energy
        self.g = relaxation

        return

    def func(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return (2.0 / PI) * l * g * w / (w**2 + g**2)

    def hta_correlations(
        self,
        distribution_func: Callable[[complex], complex],
    ) -> Tuple[list[complex], list[complex], list[complex]]:

        _c = -2.0j * self.l * self.g * distribution_func(-1.0j * self.g)
        _d = -self.g

        return [_c], [np.conj(_c)], [_d]
