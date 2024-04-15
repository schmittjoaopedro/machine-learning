# First set up the network.
import numpy as np

sigma = np.tanh  # activation function
W = np.array([  # weights
    [-2, 4, -1],
    [6, 0, -3]
])
b = np.array([0.1, -2.5])  # biases

# Define our input vector
x = np.array([0.3, 0.4, 0.1])

# Calculate the values by hand,
# and replace a1_0 and a1_1 here (to 2 decimal places)
# (Or if you feel adventurous, find the values with code!)
a1 = np.tanh(W @ x + b)
print(a1)
