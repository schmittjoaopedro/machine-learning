# Let's assume we want to train the network to give a NOT function,
# that is if you input 1 it returns 0, and if you input 0 it returns 1.
import numpy as np

# First we set the state of the network
sigma = np.tanh  # activation function
w1 = -5  # weight
b1 = 5  # bias


# Then we define the neuron activation.
def a1(a0):
    return sigma(w1 * a0 + b1)


# Finally let's try the network out!
# Replace x with 0 or 1 below,
print(f"a1(0) = {a1(0)}")
print(f"a1(1) = {a1(1)}")
