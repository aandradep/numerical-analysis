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
    def iterate(self, previous_step, t, *args, **kwargs):
        raise NotImplementedError()

    def solve(self, *args, **kwargs):
        simulations = self._simulations.copy()
        simulations[:, 0] = self._sde.XO

        for step in range(1, self._steps):
            simulations[:, step] = self.iterate(
                simulations[:, step - 1], step, *args, **kwargs
            )

        self._simulations = simulations
        return simulations

    def simulations_grid(self):
        return np.zeros((self._number_sim, self._steps))

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

    def iterate(self, previous_step, t):
        return (
            previous_step
            + self._sde.a(previous_step, t) * self._dt
            + self._sde.b(previous_step, t) * self.brownian_motion()
        )


class MilsteinScheme(EulerMaruyamaScheme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, previous_step, t):
        euler = EulerMaruyamaScheme.iterate(self, previous_step, t)
        milstein = (
            0.5
            * self._sde.b(previous_step, t)
            * self._sde.b_derivative(previous_step, t)
            * (self.brownian_motion() ** 2 - self._dt)
        )

        return euler + milstein


class TalayTubaro(MilsteinScheme):
    def __init__(self, function: Callable, simulation_method: str = "milstein", *args, **kwargs):
        self._function = function
        self._simulation_method = simulation_method
        super().__init__(*args, **kwargs)
    
    @property
    def function(self):
        return self._function

    @property
    def simulation_method(self):
        return self._simulation_method

    def iterate(self, previous_step, t):
        if self._simulation_method == "milstein":
            simulations = MilsteinScheme.iterate(self, previous_step, t)
        else:
            simulations = EulerMaruyamaScheme.solve(self, previous_step, t)
        
        return simulations

    def expected_value(self, *args, **kwargs):
        simulations_half_h = super().solve()
        simulations_h = super().solve()[:, np.arange(1, self._steps, 2)]
        f_half_h = self._function(simulations_half_h[:, self._steps-1], *args, **kwargs)
        f_h = self._function(simulations_h[:, (self._steps/2) - 1], *args, **kwargs)

        return 2 * f_half_h - f_h