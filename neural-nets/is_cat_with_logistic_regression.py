import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
from PIL import Image

# Load cat vs non-cat dataset
train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")

# Shape (num_examples, height, width, num_channels)
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
test_set_x_orig = np.array(test_dataset["test_set_x"][:])

# Convert from array (num_examples,.) to row vector (1, num_examples)
train_set_y_orig = np.array(train_dataset["train_set_y"][:])
train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = np.array(test_dataset["test_set_y"][:])
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

# Array of classes
classes = np.array(test_dataset["list_classes"][:])

# Shows an example of picture
plt.imshow(train_set_x_orig[2])
plt.show()

# Dimensions
num_train = train_set_x_orig.shape[0]
num_test = test_set_x_orig.shape[0]
num_pixels_side = train_set_x_orig.shape[1]

# Reshape training and test data to a matrix of shape (width * height * num_colors, num_examples)
# This shape facilitates computing forward and backward for all examples using vectorization
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Normalize data to be in the range [0,1]
# In this case it is easier to divide by 255 (max value of a pixel)
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Logistic regression model parameters
num_pixels = train_set_x.shape[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_pass(w, b, X):
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    return a


def calculate_cost(a, y):
    m = y.shape[1]  # Number of examples
    return -(np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m)


# Train model
weights = np.zeros((num_pixels, 1))
bias = 0
num_iterations = 2000
learning_rate = 0.005
cost_history = []

for i in range(num_iterations + 1):
    # Forward pass to calculate activation
    activation = forward_pass(weights, bias, train_set_x)

    # Calculate error cost
    cost = calculate_cost(activation, train_set_y_orig)
    cost_history.append(cost)

    # Backward Gradient Descent
    # Jacobian with partial derivatives of cost function with respect to weights and bias
    dCdz = activation - train_set_y_orig
    dCdw = np.dot(train_set_x, dCdz.T) / num_train
    dCdb = np.sum(dCdz) / num_train

    # Update parameters, Jacobian points in the direction of steepest ascent, so we to go
    # down the hill we subtract the Jacobian.
    weights = weights - learning_rate * dCdw
    bias = bias - learning_rate * dCdb

    if i % 100 == 0:
        print(f"Cost after iteration {i}: {cost}")

# Plot cost history
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost history for logistic regression during training")
plt.show()


def print_predictions(images_data, images_predictions, start, end, title=""):
    images = []
    predictions = []
    for i in range(start, end):
        image = copy.deepcopy(images_data[i])
        image = Image.fromarray(image)
        image = image.resize((64, 64))
        images.append(image)
        predictions.append("cat" if images_predictions[0, i] == 1 else "non-cat")
    plt.figure(figsize=(20, 3))
    plt.imshow(np.hstack(images))
    plt.suptitle(title)
    # Remove white borders around images
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for i, prediction in enumerate(predictions):
        plt.text(i * 64, 64, prediction, fontsize=12, color='red')
    plt.show()


# Predict for all training data
train_predictions = forward_pass(weights, bias, train_set_x)
train_predictions = np.rint(train_predictions)
train_accuracy = 100 - np.mean(np.abs(train_predictions - train_set_y_orig)) * 100
print(f"Train accuracy: {train_accuracy}%")
print_predictions(train_set_x_orig, train_predictions, 0, 10, "Train data 0-10")
print_predictions(train_set_x_orig, train_predictions, 10, 20, "Train data 10-20")
print_predictions(train_set_x_orig, train_predictions, 20, 30, "Train data 20-30")

# Predict for all test data
test_predictions = forward_pass(weights, bias, test_set_x)
test_predictions = np.rint(test_predictions)
test_accuracy = 100 - np.mean(np.abs(test_predictions - test_set_y_orig)) * 100
print(f"Test accuracy: {test_accuracy}%")
print_predictions(test_set_x_orig, test_predictions, 0, 10, "Test data 0-10")
print_predictions(test_set_x_orig, test_predictions, 10, 20, "Test data 10-20")
print_predictions(test_set_x_orig, test_predictions, 20, 30, "Test data 20-30")

print("Done")
