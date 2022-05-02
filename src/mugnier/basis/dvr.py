# coding: utf-8
r"""A Simple DVR Program

References
----------
.. [1] http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
"""

from typing import Callable
from mugnier.libs.backend import NDArray, np


class SineDVR:

    def __init__(self, start: float, stop: float, num: int) -> None:
        self.grid_points = np.linspace(start, stop, num)
        self.n = num
        self.length = abs(start - stop)

        _i = np.arange(1, self.n + 1)[:, np.newaxis]
        _j = np.arange(1, self.n + 1)[np.newaxis, :]
        self.dvr2fbr_mat = (np.sqrt(2 / (self.n + 1)) * np.sin(_i * _j * np.pi / (self.n + 1)))  # type: NDArray
        return

    @property
    def q_mat(self) -> NDArray:
        """q in DVR basis."""
        return np.diag(self.grid_points)

    @property
    def dq_mat(self) -> NDArray:
        """d/dq in DVR basis."""
        fbr_mat = np.zeros((self.n, self.n))
        l = self.length
        n = self.n
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                if j != k:
                    fbr_mat[j - 1, k - 1] = ((j - k) % 2) * 4.0 * j * k / (j**2 - k**2) / l
        u = self.dvr2fbr_mat
        return u.T @ fbr_mat @ u

    @property
    def dq2_mat(self) -> NDArray:
        """d^2/dq^2 in DVR basis."""
        j = np.arange(1, self.n + 1)
        fbr_mat = np.diag(-(j * np.pi / self.length)**2)
        u = self.dvr2fbr_mat
        return u.T @ fbr_mat @ u

    @property
    def creation_mat(self) -> NDArray:
        return (self.q_mat - self.dq_mat) / np.sqrt(2.0)

    @property
    def annihilation_mat(self) -> NDArray:
        return (self.q_mat + self.dq_mat) / np.sqrt(2.0)

    @property
    def numberer_mat(self) -> NDArray:
        q2 = self.q_mat**2
        dq2 = self.dq2_mat
        eye = np.identity(self.n)
        return 0.5 * (q2 - dq2 - eye)

    @property
    def fock2dvr_mat(self) -> NDArray:
        _, u = np.linalg.eigh(self.numberer_mat)
        # correct the direction according to annihilation_mat
        subdiag = np.diagonal(u.T @ self.annihilation_mat @ u, offset=1)
        counter = 0
        ans = np.zeros_like(u)
        for _n, _d in enumerate(subdiag):
            if _d < 0.0:
                counter += 1
            ans[:, _n] = (-1)**counter * u[:, _n]

        return ans

    def fbr_func(self, i: int) -> Callable[[NDArray], NDArray]:
        """`i`-th FBR basis function."""
        l = self.length
        x0 = self.grid_points[0]

        def _func(_x: NDArray) -> NDArray:
            return np.where(np.logical_and(x0 < _x, _x < x0 + l),
                            np.sqrt(2.0 / l) * np.sin((i + 1) * np.pi * (_x - x0) / l), 0.0)

        return _func


if __name__ == '__main__':
    b = SineDVR(-5, 5, 1000)

    n = b.numberer_mat
    ap = b.creation_mat
    am = b.annihilation_mat
    f = b.fock2dvr_mat

    print(np.diag(f.T @ ap @ f, k=-1)[:10])
