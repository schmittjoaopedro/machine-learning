# the matrix A defines the inner product
import numpy as np

# Inner product Matrix A
A = np.array([[2, 1], [1, 4]])
x = np.array([2, 2])
y = np.array([-2, -2])


def find_angle(A, x, y):
    """Compute the angle"""
    inner_prod = x.T @ A @ y
    # Fill in the expression for norm_x and norm_y below
    norm_x = np.sqrt(x.T @ A @ x)
    norm_y = np.sqrt(y.T @ A @ y)
    alpha = inner_prod / (norm_x * norm_y)
    angle = np.arccos(alpha)
    return np.round(angle, 2)


rad_angle = find_angle(A, x, y)
print("The angle between x and y is: ", np.round(rad_angle * 180 / np.pi, 2), "degrees")
