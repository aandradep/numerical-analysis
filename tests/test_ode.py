import numpy as np

from numerical_analysis.ode.euler_methods import (
    EulerMethod,
    ExplicitTrapezoid,
    ImplicitTrapezoid,
)


def func(t, y):
    return y - t**2 + 1


def test_euler_method():
    euler = EulerMethod(func, 0, 2, 3, 0.5)
    solution = euler.solve()
    expected = np.array(
        [
            [0.0, 1.5],
            [0.66666667, 2.87037037],
            [1.33333333, 4.2654321],
            [2.0, 5.1090535],
        ]
    )

    assert np.allclose(solution, expected)


def test_explict_trapezoid():
    explicit_trapezoid = ExplicitTrapezoid(func, 0, 2, 3, 0.5)
    solution = explicit_trapezoid.solve()
    expected = np.array(
        [
            [0.0, 1.68518519],
            [0.66666667, 3.23251029],
            [1.33333333, 4.673754],
            [2.0, 5.1244983],
        ]
    )

    assert np.allclose(solution, expected)


def test_implicit_trapezoid():
    implicit_trapezoid = ImplicitTrapezoid(func, 0, 2, 3, 0.5)
    solution = implicit_trapezoid.solve()
    expected = np.array(
        [
            [0.0, 1.77777778],
            [0.66666667, 3.44444444],
            [1.33333333, 5.0],
            [2.0, 5.44444444],
        ]
    )

    assert np.allclose(solution, expected)
