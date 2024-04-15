import numpy as np


def multiply(A, B):
    """
    Multiplies two polynomials

    Arguments:
    A: Coefficients of the first polynomial
    B: Coefficients of the second polynomial

    Returns:
    C: The coefficients of A*B
    """

    # Find the coefficients of both the polynomials
    na = A.shape[0]
    nb = B.shape[0]

    # Pad the smaller array with 0s
    if na < nb:
        A = np.pad(A, (0, nb - na), 'constant', constant_values=(0, 0))
        na = nb
    elif nb < na:
        B = np.pad(B, (0, na - nb), 'constant', constant_values=(0, 0))
        nb = na

    # Initialize the output array with 0s
    nc = na + nb - 1
    C = np.zeros(nc)

    # Perform the multiplication
    # You might want to break the loop over i into two separate phases
    i = 1
    while i <= na:
        C[i - 1] = A[0:i] @ np.flip(B[0:i])
        i += 1
    i = 0
    while na + i < nc:
        C[na + i] = A[i + 1:na] @ np.flip(B[i + 1:na])
        i += 1

    # Remove any extra 0s from the back of C
    C = np.trim_zeros(C)

    return C


A = np.array([1, 2])
B = np.array([3, 4])
C_exp = np.array([3, 10, 8])
np.testing.assert_allclose(multiply(A, B), C_exp, rtol=1e-5, atol=1e-10)

A = np.array([5, 6])
B = np.array([1, 3, 5, 9])
C_exp = np.array([5, 21, 43, 75, 54])
np.testing.assert_allclose(multiply(A, B), C_exp, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(multiply(B, A), C_exp, rtol=1e-5, atol=1e-10)

print("All tests passed!")
