# Neural network to draw a curve. The curve takes one input variable,
# the amount travelled along the curve from 0 to 1, and returns 2 outputs,
# the 2D coordinates of the position of points on the curve.
#
# To help capture the complexity of the curve, we shall use two hidden
# layers in our network with 6 and 7 neurons respectively.
#
# Matrices multiplication in Python:
# - Element-wise multiplication: A = B * C
# - Row-by-column multiplication: A = B @ C
#
import matplotlib.pyplot as plt
import numpy as np

# Weight and bias initialisation.
W1, W2, W3, b1, b2, b3 = [None] * 6


# Activation function and its derivative.
def sigma(z):
    # Logistic function
    return 1 / (1 + np.exp(-z))


def d_sigma(z):
    # Derivative of the logistic function
    return np.cosh(z / 2) ** (-2) / 4


# Initialises the network structure of weights and biases with given dimension and random values.
# n1 and n2 are the number of neurons in the first and second hidden layers.
def reset_network(n1=6, n2=7, random=np.random):
    global W1, W2, W3, b1, b2, b3
    # random.randn(N, M), where NxM, N = rows, M = columns
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2


# Feeds forward each activation to the next layer (predict the output) and returns
# all weighted sums and activations. Only works for two-layer networks.
def network_function(a0):
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3


# This is the cost function of a neural network with respect to a training set.
def cost(x, y):
    return np.linalg.norm(network_function(x)[-1] - y) ** 2 / x.size


# Backpropagation functions.
# Each function calculates the Jacobian gradient of the cost function with respect to a weight or bias.

def jacobian_weights_3rd_layer(x, y):
    # First get all the activations and weighted sums at each layer of the network.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)

    # Calculate partial derivatives for the third layer:
    # dC/dW3 = dC/da3 * da3/dz3 * dz3/dW3
    dc_da3 = 2 * (a3 - y)
    da3_dz3 = d_sigma(z3)
    dz3_dW3 = a2

    # Jacobian by chain rule.
    J = (dc_da3 * da3_dz3) @ dz3_dW3.T

    # Average over all training examples.
    return J / x.size


# In this function, you will implement the jacobian for the bias.
# As you will see from the partial derivatives, only the last partial derivative is different.
# The first two partial derivatives are the same as previously.
def J_b3(x, y):
    # As last time, we'll first set up the activations.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # Next you should implement the first two partial derivatives of the Jacobian.
    # ===COPY TWO LINES FROM THE PREVIOUS FUNCTION TO SET UP THE FIRST TWO JACOBIAN TERMS===
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    # For the final line, we don't need to multiply by dz3/db3, because that is multiplying by 1.
    # We still need to sum over all training examples however.
    # There is no need to edit this line.
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def J_W2(x, y):
    # The first two lines are identical to in J_W3.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    # the next two lines implement da3/da2, first σ' and then W3.
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    # then the final lines are the same as in J_W3 but with the layer number bumped down.
    J = J * d_sigma(z2)
    J = J @ a1.T / x.size
    return J


def J_b2(x, y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def J_W1(x, y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = J @ a0.T / x.size
    return J


def J_b1(x, y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


# Training
def training_data(N=100):
    x = np.arange(0, 1, 1 / N)
    y = np.array([
        16 * np.sin(2 * np.pi * x) ** 3,
        13 * np.cos(2 * np.pi * x) - 5 * np.cos(2 * 2 * np.pi * x) - 2 * np.cos(3 * 2 * np.pi * x) - np.cos(
            4 * 2 * np.pi * x)
    ]
    ) / 20
    y = (y + 1) / 2
    x = np.reshape(x, (1, N))
    return x, y


def plot_training(x, y, iterations=10000, aggression=3.5, noise=1):
    global W1, W2, W3, b1, b2, b3
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect(1)

    xx = np.arange(0, 1.01, 0.01)
    yy = np.arange(0, 1.01, 0.01)
    X, Y = np.meshgrid(xx, yy)
    Z = ((X - 0.5) ** 2 + (Y - 1) ** 2) ** (1 / 2) / (1.25) ** (1 / 2)
    ax.imshow(Z, vmin=0, vmax=1, extent=[0, 1, 1, 0], cmap="Blues")

    ax.plot(y[0], y[1], lw=1.5, color="green")

    while iterations >= 0:
        j_W1 = J_W1(x, y) * (1 + np.random.randn() * noise)
        j_W2 = J_W2(x, y) * (1 + np.random.randn() * noise)
        j_W3 = jacobian_weights_3rd_layer(x, y) * (1 + np.random.randn() * noise)
        j_b1 = J_b1(x, y) * (1 + np.random.randn() * noise)
        j_b2 = J_b2(x, y) * (1 + np.random.randn() * noise)
        j_b3 = J_b3(x, y) * (1 + np.random.randn() * noise)

        W1 = W1 - j_W1 * aggression
        W2 = W2 - j_W2 * aggression
        W3 = W3 - j_W3 * aggression
        b1 = b1 - j_b1 * aggression
        b2 = b2 - j_b2 * aggression
        b3 = b3 - j_b3 * aggression

        if iterations % 1000 == 0:
            nf = network_function(x)[-1]
            ax.plot(nf[0], nf[1], lw=2, color="magenta", alpha=0.1)
            print(f"Iter: {iterations} ; Cost: {cost(x, y)}")
        iterations -= 1

    nf = network_function(x)[-1]
    ax.plot(nf[0], nf[1], lw=2.5, color="yellow")
    plt.show()


x_input, y_output = training_data()
reset_network(n1=6, n2=7)
plot_training(x_input, y_output, iterations=10000, aggression=7, noise=1)

reset_network(n1=6, n2=7)
plot_training(x_input, y_output, iterations=25000, aggression=7, noise=1)

reset_network(n1=50, n2=50)
plot_training(x_input, y_output, iterations=10000, aggression=3.5, noise=1)

reset_network(n1=50, n2=50)
plot_training(x_input, y_output, iterations=25000, aggression=3.5, noise=1)
