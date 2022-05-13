from numerical_analysis.ode.numerical_ode import NumericalODE
from numerical_analysis.one_variable_equations import Secant


class EulerMethod(NumericalODE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def euler_iteration(function, t, w, h):
        return w + h * function(t, w)

    def approximation_method(self, t, w):
        return self.euler_iteration(self._function, t, w, self._h)


class ImplicitTrapezoid(NumericalODE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def approximation_method(self, t, w):
        t_k = t + self._h

        def _trapezoid(x_k):
            trapezoid = (self._function(t, w) + self._function(t_k, x_k)) * 1 / 2
            return x_k - w - (self._h * trapezoid)

        return Secant(_trapezoid).solve(w)


class ExplicitTrapezoid(NumericalODE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def approximation_method(self, t, w):
        x_k = EulerMethod.euler_iteration(self._function, t, w, self._h)
        t_k = t + self._h

        return w + (self._h / 2) * (self._function(t, w) + self._function(t_k, x_k))
