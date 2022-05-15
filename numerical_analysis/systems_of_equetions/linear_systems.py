import numpy as np
import sys

from abc import ABC, abstractmethod

class LinearSystems(ABC):
    def __init__(
            self, 
            mat,
            b, 
            tol=10**-3, 
            max_iter=1000
        ):
        
        self._mat = mat
        self._b = b
        self._tol = tol
        self._max_iter = max_iter
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
        
    def _l2_norm(self, xi, x0):
        return np.sqrt(sum((xi - x0)**2))
    
    def _stopping_condition(self, xi, x0):
        norm = self._l2_norm(xi, x0)
        return norm > self._tol
    
    def _warning(self, iterations):
        if iterations==self._max_iter:
            import warnings
            message = f"Algorithm did converge in {self._max_iter} iterations"
            warnings.warn(message)
    
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
