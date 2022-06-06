from numerical_analysis.pde.finit_difference import FTCS


class BlackScholesFTCS(FTCS):
    def __init__(self, *args, **kwargs):
        self._bsm = kwargs["pde"]
        super().__init__(
            alpha=self._bsm.pde_alpha,
            beta=self._bsm.pde_beta,
            gamma=self._bsm.pde_gamma,
            *args,
            **kwargs
        )