import pytest
import numpy as np

from numerical_analysis.interpolation import lagrange_interpolation


def test_lagrange_interpolation():
    x = np.array([0, 1, 2, 5])
    y = np.array([2, 3, 12, 147])

    int1 = lagrange_interpolation(x, y, 3)
    int2 = lagrange_interpolation(x, y, 0)
    int3 = lagrange_interpolation(x, y, 5)

    assert int1 == 35
    assert int2 == 2
    assert int3 == 147
