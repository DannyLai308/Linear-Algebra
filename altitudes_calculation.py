# Student Name: Danny Lai - 400491405
# Class: Linear Algebra - 4MA3
import numpy as np

def householder(input_matrix):
    # Retrieve information of input matrix and do initial setups
    # for the householder transformation
    m_row = np.shape(input_matrix)[0]
    n_column = np.shape(input_matrix)[1]
    print(f'\nGiven a {m_row} x {n_column} input matrix (m > n):\n')
    print(f'{input_matrix}\n')

    # Create a result matrix to work on and avoid manipulating original matrix
    result_matrix = input_matrix.copy() 
    H_matrix = np.identity(m_row)
    vector = np.zeros_like(result_matrix)

    for k in range (n_column):
        # Extract each column from input matrix as a vector to wokr on
        current_column = result_matrix[k:, k].astype(float) 
        e = H_matrix[k:, k] # Get e from identity matrix for calculation of vector
        if current_column[0] > 0: # Check for sign of a for calculation of alpha
            alpha = -np.linalg.norm(current_column,2)
        else:
            alpha = np.linalg.norm(current_column,2)

        # Calculate vector using current_column vector transpose, alpha, and e
        vector[k:, k] = result_matrix[k:, k].T - alpha * e
        # Calculate beta using vector transpose and vector
        beta = vector[k:, k].T @ vector[k:, k]
        if beta == 0: # Avoid division by 0 when transforming the remaining submatrix
            continue

        for j in range (k, n_column):
            gamma = vector[k:, k].T @ result_matrix[k:, j] # Calculate gamma j
            result_matrix = result_matrix.astype(float)
            # Calculate aj
            result_matrix[k:, j] = result_matrix[k:, j] - ((2*gamma)/beta)*vector[k:, k]

    # Format the result matrix for printing
    final_matrix = np.around(result_matrix, decimals=2)
    final_matrix = np.where(final_matrix == -0.00, 0.00, final_matrix)
    # Remove the last column as it was the appended vector b used for calculation only
    final_matrix = final_matrix[:, :-1] 
    print(f'Resulting matrix:\n{final_matrix}\n') # Show result matrix after transformation

    return result_matrix

def backward_substitution(n, R_matrix):
    # Initialize vector x
    x = np.zeros(n) 
    for i in range(n-1, -1, -1):
        p = R_matrix[i, i] # Calculating pivot
        if p == 0:
            continue
        x[i] = R_matrix[i, -1] / p
        for j in range (i-1,-1,-1):
            R_matrix[j, -1] -= R_matrix[j, i] * x[i]        
    return x

# Set up initial matrix and vector for the transformation
A_matrix = np.array([[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
            [1, -1, 0, 0],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
            [0, 1, -1, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1]])

b_vector= np.array([2.95,1.74,-1.45,1.32,1.23,4.45,1.61,3.21,0.45,-2.75])
x=np.array([2.95,1.74,-1.45,1.32])

# Attach b vector to the input matrix A
input_matrix=np.concatenate((A_matrix, np.reshape(b_vector, (np.size(b_vector), 1))), axis=1)

# Get the result of matrix A after householder transformation
result_matrix = householder(input_matrix)
result_matrix = np.around(result_matrix, decimals=2)

# Ensuring size of matrix R can be used to apply backward substitution
R_matrix = result_matrix[:A_matrix.shape[1], :-1] 
# Calculate altitude values
x_hat = backward_substitution(R_matrix.shape[0], result_matrix)
x_hat = np.around(x_hat, decimals=2)
x_delta = np.subtract(x_hat, x)

print(f'x true values:\n{x}\n')
print(f'x hat values:\n{x_hat}\n')
print(f'x delta values:\n{x_delta}\n')
