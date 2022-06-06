from numerical_analysis.pde.finit_difference import FTCS, BTCS


class BlackScholesFTCS(FTCS):
    def __init__(self, *args, **kwargs):
        self._bsm = kwargs["pde"]
        super().__init__(
            alpha=self.pde_alpha,
            beta=self.pde_beta,
            gamma=self.pde_gamma,
            *args,
            **kwargs
        )

    def pde_alpha(self, X, t):
        return (self._bsm.dt / 2) * (
            ((self._bsm.sigma(X, t) ** 2 * (X**2)) / (self._bsm.dx**2))
            - ((self._bsm.risk_free * X) / self._bsm.dx)
        )

    def pde_beta(self, X, t):
        return (
            1
            - (self._bsm.dt / (self._bsm.dx**2)) * self._bsm.sigma(X, t) ** 2 * X**2
            - (self._bsm.risk_free * self._bsm.dt)
        )

    def pde_gamma(self, X, t):
        return (self._bsm.dt / 2) * (
            ((self._bsm.sigma(X, t) ** 2 * X**2) / (self._bsm.dx**2))
            + (self._bsm.risk_free * X) / self._bsm.dx
        )


class BlackScholesBTCS(BTCS):
    def __init__(self, *args, **kwargs):
        self._bsm = kwargs["pde"]
        super().__init__(
            alpha=self.pde_alpha,
            beta=self.pde_beta,
            gamma=self.pde_gamma,
            *args,
            **kwargs
        )

    def pde_alpha(self, X, t):
        return (self._bsm.dt / 2) * (
            (-(self._bsm.sigma(X, t) ** 2 * (X**2)) / (self._bsm.dx**2))
            + ((self._bsm.risk_free * X) / self._bsm.dx)
        )

    def pde_beta(self, X, t):
        return (
            1
            + (self._bsm.dt / (self._bsm.dx**2)) * self._bsm.sigma(X, t) ** 2 * X**2
            + (self._bsm.risk_free * self._bsm.dt)
        )

    def pde_gamma(self, X, t):
        return (self._bsm.dt / 2) * (
            (-(self._bsm.sigma(X, t) ** 2 * X**2) / (self._bsm.dx**2))
            - ((self._bsm.risk_free * X) / self._bsm.dx)
        )