import pandas as pd
from scipy import optimize


# Explore using the Newton-Raphson method for root finding.
#
# Consider the function below:
# f(x) = x^6/6 - 3x^4 - 2x^3/3 + 27x^2/2 + 18x - 30
#
# There are two places that this function goes through zero, i.e.
# two roots, one is near x=-4 and the other is near x=1.
#
# Recall that if we linearise about a particular point x0, we can
# ask what the value of the function is at the point x0 + δx, where
# a short distance away.
# f(x0 + δx) = f(x0) + δx f'(x0)
#
# Then, if we assume that the function goes to zero somewhere nearby,
# we can re-arrange to find how far away, i.e. assume that f(x0 + δx) = 0
# and solve for δx. This becomes,
# δx = -f(x0) / f'(x0)
#
# Since the function, f(x), is not a line, this formula will (try) to
# get closer to the root, but won't exactly hit it. But this is OK,
# because we can repeat the process from the new starting point to get
# even closer,
# x_n+1 = x_n - f(x_n) / f'(x_n)

# The function f(x)
def f(x):
    return x ** 6 / 6 - 3 * x ** 4 - 2 * x ** 3 / 3 + 27 * x ** 2 / 2 + 18 * x - 30


# The derivative of the function f(x)
def d_f(x):
    return x ** 5 - 12 * x ** 3 - 2 * x ** 2 + 27 * x + 18


def descent(x):
    print("\n\nStarting at x = ", x)
    d = {"x": [x], "f(x)": [f(x)]}
    for i in range(0, 20):
        x = x - f(x) / d_f(x)
        d["x"].append(x)
        d["f(x)"].append(f(x))

    result = pd.DataFrame(d, columns=['x', 'f(x)'])
    print(result)


# Converges for y = 0 nearest point x = -4.0
descent(-4.0)

# Problem 1 with this method, should converge to a point near to 1.99, but due to the gradient of zero close to
# 1.99 the convergence is zapped to close to -4.0
descent(1.99)

# Enter an eternal loop around x = 3.1
descent(3.1)
# our hand-craft implementation is faulty, now let's use a library
result = optimize.newton(f, 3.1)
print("Optimize result using library: ", result)
