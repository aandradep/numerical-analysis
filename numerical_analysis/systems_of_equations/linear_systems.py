import numpy as np
import sys

from abc import abstractmethod
from numerical_analysis.systems_of_equations.equations_system import EquationsSystems

class LinearSystems(EquationsSystems):
    def __init__(
            self, 
            mat,
            b, 
            *args, 
            **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self._mat = mat
        self._b = b
        self._n = self._mat.shape[0]
        
        if mat.shape[0] != mat.shape[1]:
            sys.exit("The matrix is not squared")
    
    @property
    def mat(self):
        return self._mat
    
    @property
    def b(self):
        return self._b
    
    @property
    def tol(self):
        return self._tol

    @property
    def max_iter(self):
        return self._max_iter
    
    @abstractmethod
    def iterate(self, x0, xi):
        raise NotImplementedError()
    
    def solve(self, x0):
        i = 0
        continue_iterate = True
        xi = x0.copy()
        while i < self._max_iter and continue_iterate:
            xi = self.iterate(x0, xi)
            continue_iterate = self._stopping_condition(xi, x0)
            i += 1
            x0 = xi.copy()
        
        self._warning(i)
        
        return xi

class Jacobi(LinearSystems):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def iterate(self, x0, xi):
        for i in range(self._n):
            xi[i] = 1/self._mat[i,i] * (
                sum(np.delete((x0 * - self._mat[i,]), i)) + self._b[i]
            )
                
        return xi
    
class GaussSeidel(LinearSystems):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def iterate(self, x0, xi):
        for i in range(self._n):
            xi[i] = 1/self._mat[i,i] * (
                -sum((xi * self._mat[i,])[:i]) -sum((x0 * self._mat[i,])[(i+1):]) 
                + self._b[i]
            )
        
        return xi
