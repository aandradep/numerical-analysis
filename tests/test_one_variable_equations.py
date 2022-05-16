import pytest
import math

from numerical_analysis.solution_of_equations import NewtonRaphson, Secant


def func(x):
    return math.cos(x) - x


def derivative_func(x):
    return -math.sin(x) - 1


def test_newton_raphson():
    newton = NewtonRaphson(function=func, derivative=derivative_func)
    solution = newton.solve(math.pi / 4)

    assert round(solution, 5) == 0.73909


def test_secant():
    secant = Secant(func)
    solution = secant.solve(0.1)

    assert round(solution, 5) == 0.73909
