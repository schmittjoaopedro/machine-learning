# PACKAGE
import numpy as np
from utils import *


# This is the Gaussian function.
def f(x, mu, sig):
    return np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi) / sig


# Partial derivative with respect to μ.
def dfdmu(x, mu, sig):
    return f(x, mu, sig) * ((x - mu) / sig ** 2)


# Partial derivative with respect to σ.
def dfdsig(x, mu, sig):
    return f(x, mu, sig) * ((x ** 2 - 2 * x * mu + mu ** 2 - sig ** 2) / sig ** 3)


def steepest_step(x, y, mu, sig, aggression):
    # Jacobian vector
    J = np.array([
        -2 * (y - f(x, mu, sig)) @ dfdmu(x, mu, sig),
        -2 * (y - f(x, mu, sig)) @ dfdsig(x, mu, sig)
    ])
    step = -J * aggression
    return step


# First get the heights data, ranges and frequencies
x, y = heights_data()

# Next we'll assign trial values for these.
mu = 155
sig = 6
# We'll keep a track of these, so we can plot their evolution.
p = np.array([[mu, sig]])

# Plot the histogram for our parameter guess
histogram(f, [mu, sig])
# Do a few rounds of steepest descent.
for i in range(50):
    dmu, dsig = steepest_step(x, y, mu, sig, 2000)
    mu += dmu
    sig += dsig
    p = np.append(p, [[mu, sig]], axis=0)

print(p)

# Plot the path through parameter space.
contour(f, p)
# Plot the histogram for our parameter guess
histogram(f, [mu, sig])
