from numerical_analysis.integration import NumericalIntegral


class Midpoint(NumericalIntegral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def approximation_method(self, lower, upper):
        return (upper - lower) * self._function((upper + lower) / 2)
