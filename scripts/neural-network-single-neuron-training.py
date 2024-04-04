########################################################################################
# Single Neuron Neural Network - Learning weights and bias with gradient descent
########################################################################################
import numpy as np

# First define our sigma function.
sigma = np.tanh


# Next define the feed-forward equation.
def a1(w1, b1, a0):
    z = w1 * a0 + b1
    return sigma(z)


# The individual cost function is the square of the difference between
# the network output and the training data output.
def C(w1, b1, x, y):
    return (a1(w1, b1, x) - y) ** 2


# This function returns the derivative of the cost function with respect to the weight.
def dCdw(w1, b1, x, y):
    z = w1 * x + b1
    dCda = 2 * (a1(w1, b1, x) - y)  # Partial-derivative of cost with activation (dC/da).
    dadz = 1 / np.cosh(z) ** 2  # Partial-derivative of activation with weighted sum z (da/dz).
    dzdw = x  # Partial-derivative of weighted sum z with weight (dz/dw).
    return dCda * dadz * dzdw  # Return the chain rule product (dC/da * da/dz * dz/dw)


# This function returns the derivative of the cost function with respect to the bias.
def dCdb(w1, b1, x, y):
    z = w1 * x + b1
    dCda = 2 * (a1(w1, b1, x) - y)
    dadz = 1 / np.cosh(z) ** 2
    dzdb = 1
    return dCda * dadz * dzdb


# Let's start with an unfit weight and bias.
learning_rate = 0.1
w1 = 2.3
b1 = -1.2

# Training
training = [
    # x, y
    (0, 1),
    (1, 0),
]
cost = 1e10
iter = 0
while cost > 1e-3:
    dCdw_total = 0
    dCdb_total = 0
    cost = 0
    for x, y in training:
        dCdw_total += dCdw(w1, b1, x, y)
        dCdb_total += dCdb(w1, b1, x, y)
        cost += C(w1, b1, x, y)
    dCdw_total /= len(training)
    dCdb_total /= len(training)
    cost /= len(training)
    w1 -= learning_rate * dCdw_total
    b1 -= learning_rate * dCdb_total
    print(f"iter = {iter}, w1 = {w1}, b1 = {b1}, cost = {cost}")
    iter += 1

# Predictions
print(f"x = 0 -> {a1(w1, b1, 0)}")
print(f"x = 1 -> {a1(w1, b1, 1)}")
