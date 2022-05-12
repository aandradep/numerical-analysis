import math


class FixPointMethods:
    def __init__(self, function, tol=10**-4, max_iter=1000):

        self._function = function
        self._tol = tol
        self._max_iter = max_iter

    @property
    def function(self):
        return self._function

    @property
    def tol(self):
        return self._tol

    @property
    def max_iter(self):
        return self._max_iter

    def initialize(self, x0) -> dict:
        raise NotImplementedError()

    def iterate(self, params) -> float:
        raise NotImplementedError()

    def solve(self, x0, *args, **kwargs):
        i = 0
        params = self.initialize(x0)
        continue_iterate = True
        while i < self._max_iter and continue_iterate:
            previous_x = params["x"]
            params = self.iterate(params)
            continue_iterate = self._stopping_condition(previous_x, params["x"])
            i += 1

        self._warning(i)

        return params["x"]

    def _stopping_condition(self, x, new_x):
        return abs(x - new_x) > self._tol

    def _warning(self, iterations):
        if iterations == self._max_iter:
            import warnings

            message = f"Algorithm did converge in {self._max_iter} iterations"
            warnings.warn(message)


class NewtonRaphson(FixPointMethods):
    def __init__(self, derivative, *args, **kwargs):
        self._derivative = derivative
        super().__init__(*args, **kwargs)

    @property
    def derivative(self):
        return self._derivative

    def initialize(self, x0):
        return {"x": x0}

    def iterate(self, params):
        x = params["x"] - self._function(params["x"]) / self._derivative(params["x"])
        return {"x": x}


class Secant(FixPointMethods):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, x0):
        x1 = x0 * 1.1
        return {"x0": x0, "x": x1, "q0": self._function(x0), "q1": self._function(x1)}

    def iterate(self, params):
        x = params["x"] - params["q1"] * (params["x"] - params["x0"]) / (
            params["q1"] - params["q0"]
        )

        return {"x": x, "x0": params["x"], "q0": params["q1"], "q1": self._function(x)}
