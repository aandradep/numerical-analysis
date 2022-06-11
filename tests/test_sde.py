import math
import numpy as np
import pytest

import numerical_analysis.sde as sde

def call_payoff(X, K, DF):
        T =  X.shape[1] - 1
        return np.maximum(X[:,T] - K, 0).mean() * DF

@pytest.fixture
def inputs():
    gbm_sv = sde.GBMStochasticVol(mu=0.05, sigma0=0.2, sigma1=0.1, X0=10, T=1, K=10)

    return {"gbm_sv": gbm_sv}

def test_expected_value_euler(inputs):
    gbm_sv = inputs["gbm_sv"]
    gbm_sv_euler = sde.EulerMaruyamaScheme(sde=gbm_sv, dt = 1/252, number_sim=1000)
    value = gbm_sv_euler.expected_value(call_payoff, K=gbm_sv.K, DF=np.exp(-0.05))

    assert value < 1.4 and value > 1.0

def test_expected_value_milstein(inputs):
    gbm_sv = inputs["gbm_sv"]
    gbm_sv_milstein = sde.MilsteinScheme(sde=gbm_sv, dt = 1/252, number_sim=1000)
    value = gbm_sv_milstein.expected_value(call_payoff, K=gbm_sv.K, DF=np.exp(-0.05))

    assert value < 1.4 and value > 1.0
