import numpy as np
from scipy import stats


def linfit(xdat, ydat):
    xbar = np.sum(xdat) / len(xdat)
    ybar = np.sum(ydat) / len(ydat)

    # m is gradient
    m = np.sum((xdat - xbar) * ydat) / np.sum((xdat - xbar) ** 2)

    # c is y-intercept
    c = ybar - m * xbar

    # Return your values as [m, c]
    return [m, c]

# Calculate manually
x = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
y = np.array([0.1, 0.25, 0.55, 0.75, 0.85])
print(linfit(x, y))

# Using library
regression = stats.linregress(x, y)
print(regression)
