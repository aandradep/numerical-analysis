import sys
import numpy as np

from typing import Optional
from abc import abstractmethod

from numerical_analysis.pde.pde_equations import PDE
from numerical_analysis.systems_of_equations.linear_systems import SOR


class FinitDifference:
    def __init__(
        self, pde: PDE, alpha: np.vectorize, beta: np.vectorize, gamma: np.vectorize
    ):
        self._pde = pde
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._grid = self.create_grid()
        self._x_range = np.arange(0, self._pde.max_x + self._pde.dx, self._pde.dx)

    @property
    def pde(self):
        return self._pde

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def grid(self):
        return self._grid

    def create_grid(self):
        time_steps = int(self._pde.max_time / self._pde.dt)
        x_steps = int(self._pde._max_x / self._pde.dx)
        grid = np.zeros((time_steps + 1, x_steps + 1))

        grid[0, :] = self._pde.initial_condition(
            np.arange(0, self._pde.max_x + self._pde.dx, self._pde.dx)
        )
        grid[:, 0] = self._pde.boundary_conditions[0]
        grid[:, x_steps] = self._pde.boundary_conditions[1]

        return grid

    @abstractmethod
    def iterate(self, grid, i):
        raise NotImplementedError()

    def solve(self):
        grid = self._grid.copy()
        time_steps = grid.shape[0]

        for i in range(1, time_steps):
            grid[i, :] = self.iterate(grid, i)

        self._grid = grid

        return grid

    def index_grid(self, t: Optional[int] = None, x: Optional[int] = None):
        time_range = np.arange(0, self._pde.max_time + self._pde.dt, self._pde.dt)

        t_index = np.where(time_range == t)[0] if t else None
        x_index = np.where(self._x_range == x)[0] if x else None

        if t and x:
            return self._grid[t_index, x_index]
        elif x:
            return self._grid[:, x_index]
        else:
            return self._grid[t_index, :]


class FTCS(FinitDifference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if self._lambda > 0.5:
        #        sys.exit(
        #            "dt, dx and alpha are define in a way that the algorith isn't stable"
        #        )

    def iterate(self, grid, i):
        x_steps = grid.shape[1] - 1
        previous_x = self._x_range[1:x_steps]
        grid[i, 1:x_steps] = (
            self._beta(previous_x, i) * grid[i - 1, 1:x_steps]
            + self._gamma(previous_x, i) * grid[i - 1, 2 : (x_steps + 1)]
            + self._alpha(previous_x, i) * grid[i - 1, 0 : (x_steps - 1)]
        )

        return grid[i, :]


class BTCS(FinitDifference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, grid, i, omega=1.5):
        x_steps = grid.shape[1] - 1
        x0 = grid[i - 1, 1:x_steps]
        grid[i, 1:x_steps] = SOR(
            mat=self._A_matrix(x_steps, self._x_range, i), b=x0, omega=omega
        ).solve(x0)

        return grid[i, :]

    def _A_matrix(self, x_steps, x, i):
        diag = np.diag(self._beta(x, i)[: x_steps - 1])
        upper_offdiag = np.diag(self._gamma(x, i)[: x_steps - 2], 1)
        lower_offdiag = np.diag(self._alpha(x, i)[: x_steps - 2], -1)

        return diag + upper_offdiag + lower_offdiag


class CrankNicolson(FinitDifference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, grid, i, omega=1.5):
        x_steps = grid.shape[1] - 1
        b = self._B_matrix(x_steps).dot(grid[i - 1, 1:x_steps])
        grid[i, 1:x_steps] = SOR(mat=self._A_matrix(x_steps), b=b, omega=omega).solve(b)

        return grid[i, :]

    def _A_matrix(self, x_steps):
        diag = np.diag(np.repeat(1 + self._beta, x_steps - 1))
        upper_offdiag = np.diag(np.repeat(-self._gamma / 2, x_steps - 2), 1)
        lower_offdiag = np.diag(np.repeat(-self._alpha / 2, x_steps - 2), -1)

        return diag + upper_offdiag + lower_offdiag

    def _B_matrix(self, x_steps):
        diag = np.diag(np.repeat(1 - self._beta, x_steps - 1))
        upper_offdiag = np.diag(np.repeat(self._gamma / 2, x_steps - 2), 1)
        lower_offdiag = np.diag(np.repeat(self._alpha / 2, x_steps - 2), -1)

        return diag + upper_offdiag + lower_offdiag
