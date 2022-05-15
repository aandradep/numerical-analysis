import sys
import numpy as np

from abc import abstractmethod
from numerical_analysis.systems_of_equations.linear_systems import GaussSeidel
from numerical_analysis.systems_of_equations.equations_system import EquationsSystems

class NonLinearSystems(EquationsSystems):
    def __init__(
            self, 
            F_mat,
            *args, 
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self._F_mat = F_mat
        
    @property
    def F_mat(self):
        return self._F_mat
    


class Newtons(NonLinearSystems):
    def __init__(self, jacobian, *args, **kwargs):
        self._jacobian = jacobian
        super().__init__(*args, **kwargs)
    
    @property
    def jacobian(self):
        return self._jacobian

    def solve(self, X):
        i = 0
        continue_iterate = True
        
        while i < self._max_iter and continue_iterate:
            Fx = self._F_mat(X)
            Jx = self._jacobian(X)

            gauss_seidel = GaussSeidel(Jx, -Fx)
            solution = gauss_seidel.solve(X)
            Xi = solution + X
            continue_iterate = self._stopping_condition(Xi, X)
            i += 1
            X = Xi.copy()
        
        self._warning(i)
        
        return Xi