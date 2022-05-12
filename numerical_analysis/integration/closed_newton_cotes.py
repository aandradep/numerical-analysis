from numerical_analysis.integration import NumericalIntegral


class Trapezoid(NumericalIntegral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def approximation_method(self, lower, upper):
        return (upper - lower) * (self._function(lower) + self._function(upper)) / 2


class Simpsons(NumericalIntegral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def approximation_method(self, lower, upper):
        mid_point = (upper + lower) / 2
        return ((upper - lower) / 6) * (
            self._function(lower)
            + 4 * self._function(mid_point)
            + self._function(upper)
        )


class Booles(NumericalIntegral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _xi(self, lower, i, h):
        return lower + i * h

    def approximation_method(self, lower, upper):
        h = (upper - lower) / 4
        fs = [self._function(self._xi(lower, i, h)) for i in range(5)]
        return ((2 * h) / 45) * (
            (fs[0] + fs[4]) * 7 + (fs[1] + fs[3]) * 32 + fs[2] * 12
        )
