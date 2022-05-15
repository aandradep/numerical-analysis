from numerical_analysis.ode.numerical_ode import NumericalODE

class RungeKutta(NumericalODE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def approximation_method(self, t, w):
        k1 = self._h * self._function(t, w)
        k2 = self._h * self._function(t + (self._h / 2), w + (k1 / 2))
        k3 = self._h * self._function(t + (self._h / 2), w + (k2 / 2))
        k4 = self._h * self._function(t + self._h, w + k3)

        return  w + ((k1 + 2*(k2 + k3) + k4) / 6)