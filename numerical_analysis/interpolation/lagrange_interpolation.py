import numpy as np


def lagrange_interpolation(xs, ys, xi):
    n = len(xs)
    res = 0.0

    for i in range(n):
        term = ys[i]

        for j in range(n):
            if j != i:
                term = term * (xi - xs[j]) / (xs[i] - xs[j])

        res += term

    return res
