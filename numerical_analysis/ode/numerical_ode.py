import numpy as np

from abc import ABC, abstractmethod

class NumericalODE(ABC):
    def __init__(self, function, lower, upper, steps, initial_condition):
        self._function = function
        self._lower = lower
        self._upper = upper
        self._steps = steps
        self._initial_condition = initial_condition
        self._h = (upper - lower) / steps

    @property
    def function(self):
        return self._function

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def steps(self):
        return self._steps

    @property
    def initial_condition(self):
        return self._initial_condition

    @abstractmethod
    def approximation_method(self, t, w):
        raise NotImplementedError()

    def solve(self):
        w = self._initial_condition
        t = self._lower
        ys = [w]
        xs = [t]

        for i in range(self._steps):
            t = self._lower + i * self._h
            w = self.approximation_method(t, w)
            ys.append(w)
            xs.append(t + self._h)
        
        return np.column_stack((xs, ys))
