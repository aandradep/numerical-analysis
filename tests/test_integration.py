import math

from numerical_analysis.integration import Trapezoid, Simpsons, Booles, Midpoint


def _func(x):
    return math.sqrt(1 + x**2)


def test_trapezoid():
    trapezoid = Trapezoid(_func, 0, 2, 0.01)
    integral = round(trapezoid.integrate(), 3)

    assert integral == 2.958


def test_simpsons():
    simpsons = Simpsons(_func, 0, 2, 0.01)
    integral = round(simpsons.integrate(), 3)

    assert integral == 2.958


def test_booles():
    booles = Booles(_func, 0, 2, 0.01)
    integral = round(booles.integrate(), 3)

    assert integral == 2.958


def test_midpoint():
    midpoint = Midpoint(_func, 0, 2, 0.01)
    integral = round(midpoint.integrate(), 3)

    assert integral == 2.958
