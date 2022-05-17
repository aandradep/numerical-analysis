import math
import numpy as np
import pytest

from numerical_analysis.pde import CrankNicolson, BTCS, FTCS, FinitDifference


@pytest.fixture
def inputs():
    function = np.vectorize(lambda x: math.sin(math.pi * x))

    return {"function": function}


def test_finit_difference(inputs):
    fd = FinitDifference(1.0, 1.0, 0.5, 0.5, 0.5, (10, 10), inputs["function"])
    result = fd.create_grid()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 0.0, 10.0], [10.0, 0.0, 10.0]])

    assert np.allclose(result, expected)


def test_ftcs(inputs):
    ftcs = FTCS(1.0, 1.0, 0.5, 0.5, 0.5, (10, 10), inputs["function"])

    result = ftcs.solve()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])

    assert np.allclose(result, expected)

    ftcs = FTCS(1.0, 1.0, 1.0, 0.1, 0.0005, (0, 0), inputs["function"])
    ftcs.solve()
    result = ftcs.index_grid(t=0.5)
    expected = np.array(
        [
            [
                0.0,
                0.00228652,
                0.00434922,
                0.00598619,
                0.00703719,
                0.00739934,
                0.00703719,
                0.00598619,
                0.00434922,
                0.00228652,
                0.0,
            ]
        ]
    )

    assert np.allclose(result, expected)


def test_btcs(inputs):
    btcs = BTCS(1.0, 1.0, 0.5, 0.5, 0.5, (0, 0), inputs["function"])
    result = btcs.solve()
    expected = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.49975586, 0.0], [0.0, 0.25012195, 0.0]]
    )

    assert np.allclose(result, expected)

    btcs = BTCS(1.0, 1.0, 1.0, 0.1, 0.01, (0, 0), inputs["function"])
    btcs.solve()
    result = btcs.index_grid(t=0.5)
    expected = np.array(
        [
            [
                0.0,
                0.00285615,
                0.005363,
                0.00733564,
                0.00855342,
                0.00899017,
                0.00848975,
                0.00729119,
                0.00523461,
                0.00293977,
                0.0,
            ]
        ]
    )

    assert np.allclose(result, expected)


def test_crank_nicolson(inputs):
    crank_nicolson = CrankNicolson(1.0, 1.0, 1.0, 0.1, 0.01, (0, 0), inputs["function"])
    crank_nicolson.solve()
    result = crank_nicolson.index_grid(t=0.5)
    expected = np.array(
        [
            [
                0.0,
                0.00231826,
                0.00438468,
                0.00602165,
                0.00707747,
                0.0074481,
                0.00709388,
                0.00604495,
                0.00438122,
                0.00226464,
                0.0,
            ]
        ]
    )

    assert np.allclose(result, expected)
