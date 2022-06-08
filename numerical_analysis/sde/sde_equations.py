import math
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable


class SDE(ABC):
    def __init__(
        self,
        a: Callable,
        b: Callable,
        X0: float,
        T: float,
        t0: float = 0,
    ):
        self._a = a
        self._b = b
        self._X0 = X0
        self._t0 = t0
        self._T = T

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def X0(self):
        return self._X0

    @property
    def T(self):
        return self._T

    @property
    def t0(self):
        return self._t0

    @abstractmethod
    def b_derivative(self, X, t):
        return NotImplementedError()


class GBM(SDE):
    def __init__(self, mu: float, sigma: float, *args, **kwargs):
        self._mu = mu
        self._sigma = sigma
        super().__init__(a=self.mu_func, b=self.sigma_func, *args, **kwargs)

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    def mu_func(self, X, t):
        return self._mu * X

    def sigma_func(self, X, t):
        return self._sigma * X

    def b_derivative(self, X, t):
        return self._sigma


class GBMStochasticVol(SDE):
    def __init__(
        self, mu: float, sigma0: float, sigma1: float, K: float, *args, **kwargs
    ):
        self._mu = mu
        self._sigma0 = sigma0
        self._sigma1 = sigma1
        self._K = K
        super().__init__(a=self.mu_func, b=self.sigma_func, *args, **kwargs)

    @property
    def mu(self):
        return self._mu

    @property
    def sigma0(self):
        return self._sigma0

    @property
    def sigma1(self):
        return self._sigma1

    @property
    def K(self):
        return self._K

    def mu_func(self, X, t):
        return self._mu * X

    def sigma_func(self, X, t):
        sigma = self._sigma0 + self._sigma1 * np.cos(
            (2 * math.pi * X) / self._K
        ) * np.sin((2 * math.pi * t) / self._K)
        return sigma * X

    def b_derivative(self, X, t):
        return (
            -self._sigma1
            * np.sin((2 * math.pi * t) / self._K)
            * np.sin((2 * math.pi * X) / self._K)
            * ((2 * math.pi) / self._K)
        ) * X + self.sigma_func(X,t)
