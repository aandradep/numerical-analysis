import math
import pytest
import numpy as np

from numerical_analysis.systems_of_equations import Newtons

@pytest.fixture
def inputs():
    def _func1(x1, x2, x3):
       return 3 * x1 - math.cos(x2 * x3) - 0.5

    def _derivatives1(x1, x2, x3):
        return np.array([3, x3 * math.sin(x2 * x3), x2 * math.sin(x2 * x3)])
       
    def _func2(x1, x2, x3):
        return x1**2 - 81 * (x2 + 0.1)**2 + math.sin(x3) + 1.06
    
    def _derivatives2(x1, x2, x3):
        return np.array([2*x1, -162 * (x2 + 0.1), math.cos(x3)])
    
    def _func3(x1, x2, x3):
        return math.exp(-x1 * x2) + 20 * x3 + ((10 * math.pi - 3) / 3)
    
    def _derivatives3(x1, x2, x3):
        return np.array([-x2* math.exp(- x1 * x2), -x1* math.exp(- x1 * x2), 20])

    def F_mat(X):
        return np.array(
            [
                _func1(X[0], X[1], X[2]), 
                _func2(X[0], X[1], X[2]), 
                _func3(X[0], X[1], X[2])
            ]
        )

    def jacobian(X):
        j1 = _derivatives1(X[0], X[1], X[2])
        j2 = _derivatives2(X[0], X[1], X[2])
        j3 = _derivatives3(X[0], X[1], X[2])

        return np.vstack([j1, j2, j3])
    
    X = np.array([0.1, 0.1, -0.1])

    return {"F_mat": F_mat, "J": jacobian, "X": X}

def test_newtons(inputs):
    newtons = Newtons(F_mat=inputs["F_mat"], jacobian=inputs["J"])
    solution = newtons.solve(inputs["X"])
    expected = np.array([0.5000, -0.00, -0.5236])

    assert np.allclose(solution.round(4), expected)