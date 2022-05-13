import numpy as np


class NumericalIntegral:
    def __init__(self, function, lower, upper, interval_size):
        self._function = function
        self._lower = lower
        self._upper = upper
        self._interval_size = interval_size
        self._integration_ranges = np.arange(
            self._lower, self._upper + self._interval_size, self._interval_size
        )

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
    def interval_size(self):
        return self._interval_size

    @property
    def integration_ranges(self):
        return self._integration_ranges

    def approximation_method(self, lower, upper):
        raise NotImplementedError()

    def integrate(self, *args, **kwargs):
        integral = 0.0

        for i in range(1, len(self._integration_ranges)):
            a = self._integration_ranges[i - 1]
            b = self._integration_ranges[i]
            integral += self.approximation_method(a, b, *args, **kwargs)

        return integral
