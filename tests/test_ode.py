import numpy as np

from numerical_analysis.ode.euler_methods import (
    EulerMethod,
    ExplicitTrapezoid,
    ImplicitTrapezoid,
)
from numerical_analysis.ode.runge_kutta import RungeKutta


def func(t, y):
    return y - t ** 2 + 1


def test_euler_method():
    euler = EulerMethod(func, 0, 2, 3, 0.5)
    solution = euler.solve()
    expected = np.array(
        [
            [0.0, 0.5],
            [0.66666667, 1.5],
            [1.33333333, 2.87037037],
            [2.0, 4.2654321],
        ]
    )

    assert np.allclose(solution, expected)


def test_explict_trapezoid():
    explicit_trapezoid = ExplicitTrapezoid(func, 0, 2, 3, 0.5)
    solution = explicit_trapezoid.solve()
    expected = np.array(
        [
            [0.0, 0.5],
            [0.66666667, 1.68518519],
            [1.33333333, 3.23251029],
            [2.0, 4.673754],
        ]
    )

    assert np.allclose(solution, expected)


def test_implicit_trapezoid():
    implicit_trapezoid = ImplicitTrapezoid(func, 0, 2, 3, 0.5)
    solution = implicit_trapezoid.solve()
    expected = np.array(
        [[0.0, 0.5], [0.66666667, 1.77777778], [1.33333333, 3.44444444], [2.0, 5.0]]
    )

    assert np.allclose(solution, expected)


def test_runge_kutta():
    runge_kuta = RungeKutta(func, 0, 2, 3, 0.5)
    solution = runge_kuta.solve()
    expected = np.array(
        [
            [0.0, 0.5],
            [0.66666667, 1.80178326],
            [1.33333333, 3.54192563],
            [2.0, 5.29399973],
        ]
    )

    assert np.allclose(solution, expected)
