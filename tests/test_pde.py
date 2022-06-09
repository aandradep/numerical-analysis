import math
import numpy as np
import pytest

import numerical_analysis.pde as pde

@pytest.fixture
def inputs():
    function = np.vectorize(lambda x: math.sin(math.pi * x))
    lambda_fun = lambda x, i: 0.5
    lambda_btcs = lambda x, i: np.repeat(0.5, x.size)
    heat_equation = pde.HeatEquation(
        alpha=0.5,
        max_time=1.0, 
        max_x=1.0, 
        dx=0.5, 
        dt=0.5, 
        boundary_conditions=(10, 10), 
        initial_condition=function)
    
    strike = 10
    upper_bound = 20-10*0.9
    call_payoff = np.vectorize(lambda x: np.maximum(x-strike, 0))

    return {
        "function": function, 
        "pde": heat_equation, 
        "lambda": lambda_fun, 
        "lambda_btcs": lambda_btcs,
        "strike": strike, 
        "upper_bound": upper_bound, 
        "call_payoff": call_payoff
    }

def test_finit_difference(inputs):
    fd = pde.FinitDifference(pde=inputs["pde"], alpha=0.5, beta=0.5, gamma=0.5)
    result = fd.create_grid()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 0.0, 10.0], [10.0, 0.0, 10.0]])

    assert np.allclose(result, expected)


def test_ftcs(inputs):
    ftcs = pde.FTCS(pde=inputs["pde"], alpha=inputs["lambda"], beta=inputs["lambda"], gamma=inputs["lambda"])

    result = ftcs.solve()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 10.5, 10.0], [10.0, 15.25, 10.0]])

    assert np.allclose(result, expected)

def test_btcs(inputs):
    btcs = pde.BTCS(pde=inputs["pde"], alpha=inputs["lambda_btcs"], beta=inputs["lambda_btcs"], gamma=inputs["lambda_btcs"])

    result = btcs.solve()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 1.99975586, 10.0], [10.0, 3.99975583, 10.0]])

    assert np.allclose(result, expected)

def test_heat_eq_ftcs(inputs):
    result = pde.HeatEquationFTCS(pde=inputs["pde"]).solve()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])

    assert np.allclose(result, expected)

    heat_equation = pde.HeatEquation(
        alpha=1.0,
        max_time=1.0, 
        max_x=1.0, 
        dx=0.1, 
        dt=0.0005,
        boundary_conditions=(0, 0),
        initial_condition=inputs["function"])

    heat_ftcs= pde.HeatEquationFTCS(pde=heat_equation)
    heat_ftcs.solve()
    result = heat_ftcs.index_grid(t=0.5)
    expected = np.array(
        [
            [
                0.0,
                0.00228652,
                0.00434922,
                0.00598619,
                0.00703719,
                0.00739934,
                0.00703719,
                0.00598619,
                0.00434922,
                0.00228652,
                0.0,
            ]
        ]
    )

    assert np.allclose(result, expected)

def test_heat_eq_btcs(inputs):
    heat_equation = pde.HeatEquation(
        alpha=1.0,
        max_time=1.0, 
        max_x=1.0, 
        dx=0.1, 
        dt=0.01,
        boundary_conditions=(0, 0),
        initial_condition=inputs["function"])
    heat_btcs = pde.HeatEquationBTCS(pde=heat_equation)
    heat_btcs.solve()

    result = heat_btcs.index_grid(t=0.5)
    expected = np.array(
        [
            [
                0.0,
                0.00285615,
                0.005363,
                0.00733564,
                0.00855342,
                0.00899017,
                0.00848975,
                0.00729119,
                0.00523461,
                0.00293977,
                0.0,
            ]
        ]
    )

    assert np.allclose(result, expected)
    
def test_crank_nicolson(inputs):
    cn = pde.CrankNicolson(pde=inputs["pde"], alpha=0.5, beta=0.5, gamma=0.5)

    result = cn.solve()
    expected = np.array([[10.0, 1.0, 10.0], [10.0, 0.33300781, 10.0], [10.0, 0.11121941, 10.0]])

    assert np.allclose(result, expected)

def test_heat_eq_crank_nicolson(inputs):
    heat_equation = pde.HeatEquation(
        alpha=1.0,
        max_time=1.0, 
        max_x=1.0, 
        dx=0.1, 
        dt=0.01,
        boundary_conditions=(0, 0),
        initial_condition=inputs["function"])
    heat_cn = pde.HeatEquationCN(pde=heat_equation)
    heat_cn.solve()
    result = heat_cn.index_grid(t=0.5)
    expected = np.array(
        [
            [
                0.0,
                0.00231826,
                0.00438468,
                0.00602165,
                0.00707747,
                0.0074481,
                0.00709388,
                0.00604495,
                0.00438122,
                0.00226464,
                0.0,
            ]
        ]
    )

    assert np.allclose(result, expected)

def test_black_scholes_ftcs(inputs):
    bsm = pde.BlackScholes(
        strike=inputs["strike"],
        sigma=0.3**2,
        risk_free=0.1,
        max_time=2, 
        max_x=20, 
        dx=0.4, 
        dt=1/252,
        boundary_conditions=(0, inputs["upper_bound"]),
        initial_condition=inputs["call_payoff"]
    )

    bsm_ftcs = pde.BlackScholesFTCS(pde=bsm)
    bsm_ftcs.solve()

    result = bsm_ftcs.index_grid(t=0.5, x=14)
    assert round(result[0], 2) == 4.61

def test_black_scholes_btcs(inputs):
    bsm = pde.BlackScholes(
        strike=inputs["strike"],
        sigma=0.3**2,
        risk_free=0.1,
        max_time=2, 
        max_x=20, 
        dx=2, 
        dt=20/252,
        boundary_conditions=(0, inputs["upper_bound"]),
        initial_condition=inputs["call_payoff"]
    )

    bsm_btcs = pde.BlackScholesBTCS(pde=bsm)
    bsm_btcs.solve()

    result = bsm_btcs.index_grid(t=140/252, x=14)
    assert round(result[0], 2) == 4.45

def test_black_scholes_sv_ftcs(inputs):
    bsm_sv = pde.BSMStochasticVol(
        strike=inputs["strike"],
        sigma0=0.2**2,
        sigma1=0.1**2,
        risk_free=0.1,
        max_time=2,
        max_x=20,
        dx=0.4,
        dt=1/252,
        boundary_conditions=(0, inputs["upper_bound"]),
        initial_condition=inputs["call_payoff"]
    )
    sv_ftcs = pde.BlackScholesFTCS(pde=bsm_sv)
    sv_ftcs.solve()
    result = sv_ftcs.index_grid(t=0.5, x=14)
    assert round(result[0], 2) == 4.51