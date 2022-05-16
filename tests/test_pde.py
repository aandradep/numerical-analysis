import math
import numpy as np

from numerical_analysis.pde import FTCS


def test_ftcs():
    ftcs = FTCS(
        1.0, 1.0, 0.5, 0.5, 0.5, (10, 10), np.vectorize(lambda x: math.sin(math.pi * x))
    )

    initial_grid = ftcs.create_grid()
    expected_initial_grid = np.array(
        [[10.0, 1.0, 10.0], [10.0, 0.0, 10.0], [10.0, 0.0, 10.0]]
    )

    assert np.allclose(initial_grid, expected_initial_grid)

    result = ftcs.solve()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])

    assert np.allclose(result, expected)

    ftcs = FTCS(
        1.0,
        1.0,
        1.0,
        0.1,
        0.0005,
        (0, 0),
        np.vectorize(lambda x: math.sin(math.pi * x)),
    )
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
