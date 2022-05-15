import warnings
import numpy as np
           

class EquationsSystems:
    def __init__(
            self, 
            tol=10**-3, 
            max_iter=1000
        ):
        
        self._tol = tol
        self._max_iter = max_iter
    
    @property
    def tol(self):
        return self._tol

    @property
    def max_iter(self):
        return self._max_iter
        
    def _l2_norm(self, xi, x0):
        return np.sqrt(sum((xi - x0)**2))
    
    def _stopping_condition(self, xi, x0):
        norm = self._l2_norm(xi, x0)
        return norm > self._tol
    
    def _warning(self, iterations):
        if iterations==self._max_iter:
            message = f"Algorithm did converge in {self._max_iter} iterations"
            warnings.warn(message)
