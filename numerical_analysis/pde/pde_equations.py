import math
import numpy as np

from typing import Tuple
from abc import ABC


class PDE(ABC):
    def __init__(
        self,
        max_time: float,
        max_x: float,
        dx: float,
        dt: float,
        boundary_conditions: Tuple[float, float],
        initial_condition: np.vectorize,
    ):

        self._max_time = max_time
        self._max_x = max_x
        self._dx = dx
        self._dt = dt
        self._boundary_conditions = boundary_conditions
        self._initial_condition = initial_condition

    @property
    def max_time(self):
        return self._max_time

    @property
    def max_x(self):
        return self._max_x

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


class HeatEquation(PDE):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._lambda = self._alpha**2 * (self._dt / self._dx**2)

    @property
    def alpha(self):
        return self._alpha

    @property
    def lambda_(self):
        return self._lambda


class BlackScholes(PDE):
    def __init__(self, strike: float, sigma: float, risk_free: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._strike = strike
        self._sigma = sigma
        self._risk_free = risk_free

    @property
    def strike(self):
        return self._strike

    @property
    def risk_free(self):
        return self._risk_free

    def sigma(self, S, t):
        return self._sigma


class BSMStochasticVol(PDE):
    def __init__(
        self,
        strike: float,
        sigma0: float,
        sigma1: float,
        risk_free: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._strike = strike
        self._sigma0 = sigma0
        self._sigma1 = sigma1
        self._risk_free = risk_free

    @property
    def strike(self):
        return self._strike

    @property
    def sigma0(self):
        return self._sigma0

    @property
    def sigma0(self):
        return self._sigma1

    @property
    def risk_free(self):
        return self._risk_free

    def sigma(self, S, t):
        return self._sigma0 + self._sigma1 * (
            math.cos((2 * math.pi * t) / self._max_time)
            * np.exp(-(((S / self._strike) - 1) ** 2))
        )
