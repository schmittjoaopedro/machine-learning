import numpy as np
from scipy import optimize


def f(x, y):
    return np.exp(-(2 * x * x + y * y - x * y) / 2)


def g(x, y):
    return x * x + 3 * (y + 1) ** 2 - 1


# Next their partial derivatives,
def dfdx(x, y):
    return 1 / 2 * (-4 * x + y) * f(x, y)


def dfdy(x, y):
    return 1 / 2 * (x - 2 * y) * f(x, y)


def dgdx(x, y):
    return 2 * x


def dgdy(x, y):
    return 6 * (y + 1)


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


solve(-1, -1, 0)
solve(1, -1, 0)
solve(-0.5, -1.5, 0)
solve(-0.2, -0.5, 0)
