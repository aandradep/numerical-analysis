import numpy as np

from numerical_analysis.pde.finit_difference import FTCS, BTCS, CrankNicolson


class HeatEquationFTCS(FTCS):
    def __init__(self, *args, **kwargs):
        lambda_fun = np.vectorize(lambda x, t: kwargs["pde"].lambda_)
        beta_fun = np.vectorize(lambda x, t: 1 - 2 * kwargs["pde"].lambda_)
        super().__init__(
            alpha=lambda_fun, beta=beta_fun, gamma=lambda_fun, *args, **kwargs
        )


class HeatEquationBTCS(BTCS):
    def __init__(self, *args, **kwargs):
        lambda_fun = np.vectorize(lambda x: -kwargs["pde"].lambda_)
        beta_fun = np.vectorize(lambda x: 1 + 2 * kwargs["pde"].lambda_)
        super().__init__(
            alpha=lambda_fun, beta=beta_fun, gamma=lambda_fun, *args, **kwargs
        )


class HeatEquationCN(CrankNicolson):
    def __init__(self, *args, **kwargs):
        super().__init__(
            alpha=kwargs["pde"].lambda_,
            beta=kwargs["pde"].lambda_,
            gamma=kwargs["pde"].lambda_,
            *args,
            **kwargs
        )
