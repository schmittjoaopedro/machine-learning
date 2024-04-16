import numpy as np
from numpy.testing import assert_allclose


def mean(X):
    """Compute the sample mean for a dataset."""
    return np.mean(X, axis=0)


def cov(X):
    """
    Compute the sample covariance for a dataset.
    Returns:
        ndarray: ndarray with shape (D, D), the sample covariance of the dataset `X`.
    """
    N, D = X.shape
    covariance_matrix = np.zeros((D, D))
    mu = mean(X)
    mu = np.reshape(mu, newshape=(-1, 1))
    for n in range(N):
        vector = np.reshape(X[0], newshape=(-1, 1))
        covariance_matrix += (vector - mu) @ (vector - mu).T
    covariance_matrix /= N
    #covariance_matrix = np.cov(X, rowvar=False, bias=True)
    return covariance_matrix


# Test case 1
X = np.array([
    [0., 1.],
    [1., 2.],
    [0., 1.],
    [1., 2.]])
expected_cov = np.array(
    [[0.25, 0.25],
     [0.25, 0.25]])

assert_allclose(cov(X), expected_cov, rtol=1e-5)

# Test case 2
X = np.array([[0., 1.],
              [2., 3.]])
expected_cov = np.array(
    [[1., 1.],
     [1., 1.]])

assert_allclose(cov(X), expected_cov, rtol=1e-5)

# Test covariance is zero
X = np.array([[0., 1.],
              [0., 1.],
              [0., 1.]])
expected_cov = np.zeros((2, 2))

assert_allclose(cov(X), expected_cov, rtol=1e-5)


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation

    Args:
        S: `ndarray` of shape (D, D), the sample covariance matrix for some dataset.
        A, b: `ndarray` of shape (D, D) and (D,), affine transformation applied to x

    Returns:
        the sample covariance matrix of shape (D, D) after the transformation
    """
    affine_cov = A @ S @ A.T
    return affine_cov


A = np.array([[0, 1], [2, 3]])
b = np.ones(2)
m = np.full((2,), 2)
S = np.eye(2) * 2

expected_affine_cov = np.array(
    [[2., 6.],
     [6., 26.]])

assert_allclose(affine_covariance(S, A, b),
                expected_affine_cov, rtol=1e-4)
