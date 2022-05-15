import pytest
import numpy as np

from numerical_analysis.systems_of_equetions import Jacobi, GaussSeidel


@pytest.fixture
def inputs():
    mat = np.array(
        [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]], dtype=float
    )
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    return {"mat": mat, "b": b, "x0": x0}


def test_jacobi(inputs):
    jacobi = Jacobi(inputs["mat"], inputs["b"])
    solution = jacobi.solve(inputs["x0"])
    expected = np.array([0.99994242, 2.00008477, -1.00006833, 1.0001085])

    assert np.allclose(solution, expected)


def test_gauss_seidel(inputs):
    gauss_seidel = GaussSeidel(inputs["mat"], inputs["b"])
    solution = gauss_seidel.solve(inputs["x0"])
    expected = np.array([1.00009128, 2.00002134, -1.00003115, 0.9999881])

    assert np.allclose(solution, expected)
