import numpy as np


def diagonalize(A):
    """
    Diagonalizes the input matrix A

    Arguments:
    A: A two dimensional Numpy array which is guaranteed to be diagonalizable

    Returns:
    S, D, S_inv: As explained above
    """

    # Retrieve the number of rows in A
    n = A.shape[0]

    # Get the eigenvalues and eigenvectors of A
    eig_vals, S = np.linalg.eig(A)
    idx = eig_vals.argsort()[::1]
    eig_vals = eig_vals[idx]
    S = S[:, idx]

    # Start by initializing D to a matrix of zeros of the appropriate shape
    D = np.zeros(shape=(n, n))

    # Set the diagonal element of D to be the eigenvalues
    for i in range(n):
        D[i, i] = eig_vals[i]

    # Compute the inverse of S
    S_inv = np.linalg.inv(S)

    return S, D, S_inv


A = np.array([[1, 5],
              [2, 4]])
S_exp = np.array([[-0.92847669, -0.70710678],
                  [0.37139068, -0.70710678]])
D_exp = np.array([[-1, 0],
                  [0, 6]])
S_inv_exp = np.array([[-0.76930926, 0.76930926],
                      [-0.40406102, -1.01015254]])

S, D, S_inv = diagonalize(A)
np.testing.assert_allclose(S_exp, S, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(D_exp, D, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(S_inv_exp, S_inv, rtol=1e-5, atol=1e-10)

A = np.array([[4, -9, 6, 12],
              [0, -1, 4, 6],
              [2, -11, 8, 16],
              [-1, 3, 0, -1]])
S_exp = np.array([[-5.000000e-01,  8.017837e-01,  9.045340e-01,  3.779645e-01],
                  [-5.000000e-01,  5.345225e-01,  3.015113e-01,  7.559289e-01],
                  [ 5.000000e-01, -6.416060e-17,  3.015113e-01,  3.779645e-01],
                  [-5.000000e-01,  2.672612e-01, -1.164823e-15,  3.779645e-01]])
D_exp = np.array([[1, 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
S_inv_exp = np.array([[ -2.      ,  10.      ,  -4.      , -14.      ],
                      [ -3.741657,  22.449944, -11.224972, -29.933259],
                      [  3.316625, -13.266499,   6.63325 ,  16.583124],
                      [  0.      ,  -2.645751,   2.645751,   5.291503]])

S, D, S_inv = diagonalize(A)
np.testing.assert_allclose(S_exp, S, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(D_exp, D, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(S_inv_exp, S_inv, rtol=1e-5, atol=1e-10)

print("All tests passed!")
