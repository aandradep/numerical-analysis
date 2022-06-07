import math
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Callable

from numerical_analysis.sde.sde_equations import SDE


class NumericalSDE(ABC):
    def __init__(self, sde: SDE, dt:float, number_sim: float = 1000):
        self._sde = sde
        self._dt = dt
        self._number_sim = number_sim
        self._steps = self.steps
        self._simulations = self.simulations_grid()

    @property
    def sde(self):
        return self._sde

    @property
    def number_sim(self):
        return self._number_sim

    @property
    def simulations(self):
        return self._simulations

    @property
    def steps(self):
        return int((self._sde.T - self._sde.t0)/self._dt)

    @abstractmethod
    def iterate(self, previous_step, t, brownian_motion, *args, **kwargs):
        raise NotImplementedError()

    def solve(self, *args, **kwargs):
        simulations = self.simulations_grid()
        simulations[:, 0] = self._sde.XO

        for step in range(1, self._steps):
            simulations[:, step] = self.iterate(
                simulations[:, step - 1], step, self.brownian_motion(), *args, **kwargs
            )

        self._simulations = simulations
        return simulations

    def expected_value(self, function: Callable, *args, **kwargs):
        half_h_mat = self.simulations_grid()
        h_mat = self.simulations_grid(int(self._steps / 2))

        half_h_mat[:, 0] = self._sde._X0
        h_mat[:, 0] = self._sde._X0

        for step in range(1, self._steps):
            brownian_motion = self.brownian_motion()
            half_h_mat[:, step] = self.iterate(half_h_mat[:, step - 1], step, brownian_motion)

            if (step % 2) == 0:
                h_brownian_motion += brownian_motion
                h_mat[:, int(step / 2)]  = self.iterate(h_mat[:, int(step / 2) - 1], step / 2, h_brownian_motion)
                pass
            else:
                h_brownian_motion = brownian_motion

        half_h_expected_value = function(half_h_mat[:, self._steps - 1], *args, **kwargs)
        h_expected_value = function(h_mat[:, int(self._steps / 2) - 1], *args, **kwargs)

        return 2 * half_h_expected_value - h_expected_value

    def simulations_grid(self, steps: int=None):
        return np.zeros((self._number_sim, steps if steps is not None else self._steps))

    def brownian_motion(self):
        return np.random.randn(self._number_sim) * math.sqrt(self._dt)

    def plot_simulations(self, title):
        x_axis = range(1, self._steps + 1)
        [
            plt.plot(x_axis, self._simulations[step, :], alpha=0.03, color="blue")
            for step in range(0, self._number_sim)
        ]
        plt.title(title)


class EulerMaruyamaScheme(NumericalSDE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, previous_step, t, brownian_motion):
        return (
            previous_step
            + self._sde.a(previous_step, t) * self._dt
            + self._sde.b(previous_step, t) * brownian_motion
        )


class MilsteinScheme(EulerMaruyamaScheme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, previous_step, t, brownian_motion):
        euler = EulerMaruyamaScheme.iterate(self, previous_step, t, brownian_motion)
        milstein = (
            0.5
            * self._sde.b(previous_step, t)
            * self._sde.b_derivative(previous_step, t)
            * (brownian_motion ** 2 - self._dt)
        )

        return euler + milstein
