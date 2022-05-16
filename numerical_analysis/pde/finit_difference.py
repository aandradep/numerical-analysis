from abc import abstractmethod
import sys
import numpy as np

from typing import Tuple, Optional


class FinitDifference:
    def __init__(
        self,
        max_time: float,
        max_x: float,
        alpha: float,
        dx: float,
        dt: float,
        boundary_conditions: Tuple[float, float],
        initial_condition: np.vectorize,
    ):
        self._max_time = max_time
        self._max_x = max_x
        self._alpha = alpha
        self._dx = dx
        self._dt = dt
        self._boundary_conditions = boundary_conditions
        self._initial_condition = initial_condition
        self._lambda = self._alpha ** 2 * (self._dt / self._dx ** 2)
        self._grid = self.create_grid()

    @property
    def max_time(self):
        return self._max_time

    @property
    def max_x(self):
        return self._max_x

    @property
    def alpha(self):
        return self._alpha

    @property
    def dx(self):
        return self._dx

    @property
    def dt(self):
        return self._dt

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @property
    def initial_condition(self):
        return self._initial_condition

    @property
    def grid(self):
        return self._grid()

    def create_grid(self):
        time_steps = int(self._max_time / self._dt)
        x_steps = int(self._max_x / self._dx)
        grid = np.empty((time_steps + 1, x_steps + 1))

        grid[0, :] = self._initial_condition(
            np.arange(0, self._max_x + self._dx, self._dx)
        )
        grid[:, 0] = self._boundary_conditions[0]
        grid[:, x_steps] = self._boundary_conditions[1]

        return grid

    @abstractmethod
    def iterate(self, grid, i):
        raise NotImplementedError()

    def solve(self):
        grid = self._grid.copy()
        time_steps = grid.shape[0]

        for i in range(1, time_steps):
            grid[
                i,
            ] = self.iterate(grid, i)

        self._grid = grid

        return grid

    def index_grid(self, t: Optional[int] = None, x: Optional[int] = None):
        time_range = np.arange(0, self._max_time + self._dt, self._dt)
        x_range = np.arange(0, self._max_x + self._dx, self._dx)

        t_index = np.where(time_range == t)[0] if t else None
        x_index = np.where(x_range == x)[0] if x else None

        if t and x:
            return self._grid[t_index, x_index]
        elif x:
            return self._grid[:, x_index]
        else:
            return self._grid[t_index, :]


class FTCS(FinitDifference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._lambda > 0.5:
            sys.exit(
                "dt, dx and alpha are define in a way that the algorith isn't stable"
            )

    def iterate(self, grid, i):
        x_steps = grid.shape[1] - 1
        grid[i, 1:x_steps] = (1 - 2 * self._lambda) * grid[
            i - 1, 1:x_steps
        ] + self._lambda * (
            grid[i - 1, 2 : (x_steps + 1)] + grid[i - 1, 0 : (x_steps - 1)]
        )

        return grid[i,:]


class BTCS(FinitDifference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
