import math

import numpy as np

# Eigenvalues and Eigenvectors
M = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)

print("Question 1")
M = np.array([[4, -5, 6],
              [7, -8, 6],
              [3/2, -1/2, -2]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)
vectors = [
    np.array([1/math.sqrt(6), -1/math.sqrt(6), 2/math.sqrt(6)]),
    np.array([-1, 1, -2]),
    np.array([-3, -3, -1]),
    np.array([-3, -2, 1]),
    np.array([1/2, -1/2, -1]),
    np.array([-2/math.sqrt(9), -2/math.sqrt(9), 1/math.sqrt(9)]),
]
for v in vectors:
    for val in vals:
        res = (M - (val * np.identity(3))) @ v
        res = np.round(res)
        if np.array_equal(res, np.zeros(3)):
            print(v)
# [-2/sqrt(9), -2/sqrt(9), 1/sqrt(9)]
# [-3, -3, -1]
# [1/2, -1/2, -1]



print("Question 2")
M = np.array([[0, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)
# "because of the loop"
# "Other eigenvalues are not small compared to 1, and so do not decay away with each power iteration"
# NOT "Some of the eigenvectors are complex"


print("Question 3")
M = np.array([[0.1, 0.1, 0.1, 0.7],
              [0.7, 0.1, 0.1, 0.1],
              [0.1, 0.7, 0.1, 0.1],
              [0.1, 0.1, 0.7, 0.1]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)
# "there's now a probability of moving to any site" is a solution
# "The other eigenvalues get smaller."


print("Question 4")
M = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)
print(np.linalg.det(M))
# "there are loops in the system"
# "there are two eigen vectors of 1"
# "There isn't a unique PageRank."

print("Question 5")
M = np.array([[0.1, 0.7, 0.1, 0.1],
              [0.7, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.7],
              [0.1, 0.1, 0.7, 0.1]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)
# "none of the other options"

print("Question 6")
print("l^2 - 2l + 1/4")

print("Question 7")
print("l1 = 1 - sqrt(3)/2")
print("l2 = 1 + sqrt(3)/2")

print("Question 8")
A = np.array([[3/2, -1],
              [-1/2, 1/2]])
vals, vecs = np.linalg.eig(A)
e1 = (A - (vals[0] * np.identity(2)))
e2 = (A - (vals[1] * np.identity(2)))

v1 = np.array([-1-math.sqrt(3), 1])
v2 = np.array([-1+math.sqrt(3), 1])
v1 = np.round(v1)
v2 = np.round(v2)
print(e1 @ v1)
print(e2 @ v2)
# Correct: v1 = [-1 - sqrt(3), 1]
# Correct: v2 = [-1 + sqrt(3), 1]


print("Question 9")
C = np.array([v1, v2])
C_inv = np.linalg.inv(C)
D = C_inv @ A @ C
D = np.array([
    [D[0][0], 0],
    [0, D[1][1]]
])
print(D)
# Correct: [[1 + sqrt(3)/2, 0], [0, 1 - sqrt(3)/2]]

print("Question 10")
A = C @ (D @ D) @ C_inv
print(A)
# Correct: [[11/4, -2], [-1, 3/4]]
