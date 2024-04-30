import numpy as np

# let's create a matrix multiplication function
def matrix_mul(A, B):
    C = np.zeros((len(A), len(B[0])))
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

# let's create some matrices
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

# let's call the function
print(matrix_mul(A, B))

from numpy import matmul

matmul(A, B)