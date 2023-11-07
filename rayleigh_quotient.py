# Student Name: Danny Lai - 400491405
# Class: Linear Algebra - 4MA3
import numpy as np

def rayleigh_quotient(A_matrix, x0):
    n_iteration = 0
    x_previous = x0 # x_k-1 is x_previous
    I_matrix = np.identity(A_matrix.shape[0])
    error = 1.0
    # Compute shift
    sigma = (x_previous.T @ A_matrix @ x_previous) / (x_previous.T @ x_previous)

    for i in range(100):
        n_iteration += 1
        # Generate next vector
        y_k = np.linalg.inv(A_matrix - sigma * I_matrix) @ x_previous
        # Normalize
        x_k = y_k / np.linalg.norm(y_k, ord=np.inf)
        lamb = (x_previous.T @ A_matrix @ x_previous) / (x_previous.T @ x_previous)

        # Check for tolerance of 0.0001
        error = np.linalg.norm((A_matrix @ x_k - lamb * x_k) / (A_matrix @ x_k))
        if error < 0.0001:
            break
        x_previous = x_k
        sigma = lamb
    return lamb, n_iteration

A_matrix = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
              [0.3945, 2.7328, -0.3097, 0.1129],
              [0.4198, -0.3097, 2.5675, 0.6079],
              [1.1159, 0.1129, 0.6079, 1.7231]])
print(f'\nGiven a {A_matrix.shape[0]} x {A_matrix.shape[0]} matrix:')
print('\n#\tStarting Eigen Vector\t\tEigen Value\t\t Number of iterations\n')
for j in range (4):
    v0 = np.zeros(4)
    v0[j] = 1.0
    v0 = v0.reshape(-1, 1)
    eigen_val, n_iteration = rayleigh_quotient(A_matrix, v0)
    print(f'{j+1}\te{j+1}\t\t\t\t{eigen_val}\t\t {n_iteration}\n')

# QR Iteration Algorithm
def gramSchmidt(A_matrix):
    # Get the size of the n x n matrix to initialize upper triangular matrix R
    n = A_matrix.shape[0] 
    q = np.zeros((n, n))
    r = np.zeros((n, n))

    # Perform Q R factorization
    for i in range(n):
        current_column = A_matrix[:, i]
        for j in range(i):
            r[j, i] = np.dot(q[:, j], A_matrix[:, i])
            current_column -= r[j, i] * q[:, j]

        r[i, i] = np.linalg.norm(current_column)
        q[:, i] = current_column / r[i, i]

    return q, r    


def qr_iteration(A_matrix):
    for i in range (100):
        q, r = gramSchmidt(A_matrix) # Compute QR factorization
        result_matrix = np.dot(r, q) # Generate next matrix

        # Check for tolerance of 0.0001
        previous_eigen = np.around(np.diagonal(A_matrix), decimals=8)
        current_eigen = np.around(np.diagonal(result_matrix), decimals=8)
        error = np.linalg.norm((current_eigen - previous_eigen) / current_eigen)
        if error < 0.0001:
            break
        A_matrix = result_matrix
        previous_eigen = current_eigen
    return i, current_eigen

qr_iterations, eigen_val = qr_iteration(A_matrix)
print(f'The eigen values are: {eigen_val}')
print(f'The number of iterations for the convergence is {qr_iterations}')