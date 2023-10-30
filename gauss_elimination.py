# Student Name: Danny Lai - 400491405
# Class: Linear Algebra - 4MA3
import numpy as np

def gauss_elimination_algo(A_matrix, b_vector):

    # Error Checking - Checks if matrix A is a square n x n matrix and vector b is a single column vector
    # that matches n elements of matrix A  
    if A_matrix.shape[0] != A_matrix.shape[1]:
        print('Error: Matrix A must be a square n x n matrix')
        return

    if b_vector.shape[0] !=  A_matrix.shape[0] or b_vector.ndim != 1:
        print('Error: Vector contains more than 1 column or has mismatched number of elements compared to matrix A') 
        return
    
    # Initializations
    n = A_matrix.shape[0] # Get number of rows
    print(f'Given a {n} x {n} matrix')
    i = 0 # Row iteration
    I_matrix = np.identity(n, dtype=float)
    print(f'Corresponding identity matrix is: \n{I_matrix}\n')
    
    # Gauss elimination phase
    while i < n:
        if A_matrix[i][i] == 0.0:
            print('Error: Assume we do not use partial or complete pivoting, the pivoting cannot be zero')
            return
        # Iterate from the second row to n row
        for j in range(i+1, n):
            scaling = A_matrix[j][i] / A_matrix[i][i] # Current element divided by pivoting
            I_matrix[j][i] = scaling # Produce Lower triangular system of n x n matrix
            A_matrix[j] = A_matrix[j] - scaling * A_matrix[i] # Produce Upper triangular system of n x n matrix
            print(f'Elimination row {j+1}:\n{A_matrix}\n')
        i += 1
    print('-------------------------------------------\n')
    print(f'Upper triangular system of n x n matrix is: \n{A_matrix}\n')
    print(f'Lower triangular system of n x n matrix is: \n{I_matrix}\n')

    # Solve the given system
    return backward_substitution(n, A_matrix, forward_substitution(n, I_matrix, b))


def forward_substitution(n, L_matrix, b):
    # Initialize vector y
    y = np.zeros(n) 
    for i in range(n):
        y[i] = ((b[i] - np.dot(L_matrix[i, :i], y[:i])) / L_matrix[i, i])
    print(f'From L and b, we calculate the y vector as:\n{y}\n')
    return y


def backward_substitution(n, U_matrix, y):
    # Initialize vector x
    x = np.zeros(n) 
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U_matrix[i, i+1:], x[i+1:])) / U_matrix[i, i]
    print(f'From U and y, we calculate the x vector as:\n{x}:\n')

    for j in range(x.shape[0]):
        print(f'x{j+1} = {x[j]}\n')
    return x


# Assigned matrix and vector
A = np.array([[1, 3, 0],
             [0, 1, -3],
             [2, 0, 1]], dtype=float)

b = np.array([-2, 2, 1,], dtype=float)

x = gauss_elimination_algo(A,b)

