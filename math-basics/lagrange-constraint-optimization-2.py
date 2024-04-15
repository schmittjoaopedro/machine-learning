import numpy as np
from scipy import optimize


def f(x, y):
    return -np.exp(x - y ** 2 + x * y)


def g(x, y):
    return np.cosh(y) + x - 2


# Next their partial derivatives,
def dfdx(x, y):
    return (y + 1) * f(x, y)


def dfdy(x, y):
    return (x - 2 * y) * f(x, y)


def dgdx(x, y):
    return 1


def dgdy(x, y):
    return np.sinh(y)


# Lagrange gradient function for parameters x, y, and λ,
# and partial derivatives of function (f) and constraint(g)
# with respect to x and y.
def DL(xyλ):
    [x, y, λ] = xyλ
    return np.array([
        dfdx(x, y) - λ * dgdx(x, y),
        dfdy(x, y) - λ * dgdy(x, y),
        - g(x, y)
    ])


def solve(x0, y0, λ0):
    # Use Newton-Raphson method to find the roots of the system of equations
    x, y, λ = optimize.root(DL, [x0, y0, λ0]).x
    print("x = %g" % x)
    print("y = %g" % y)
    print("λ = %g" % λ)
    print("f(x, y) = %g" % f(x, y))
    print("g(x, y) = %g\n" % g(x, y))  # Must solve to zero as per the constraint


solve(0, 0, 0)
