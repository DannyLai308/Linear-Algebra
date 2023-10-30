import numpy as np

def householder(input_matrix):
    # Retrieve information of input matrix and do initial setups
    # for the householder transformation
    m_row = np.shape(input_matrix)[0]
    n_column = np.shape(input_matrix)[1]
    print(f'\nGiven a {m_row} x {n_column} matrix (m > n):\n')
    print(f'{input_matrix}\n')

    # Create a result matrix to work on and avoid manipulating original matrix
    result_matrix = input_matrix.copy() 
    H_matrix = np.identity(m_row)
    vector = np.zeros_like(result_matrix)

    for k in range (n_column):
        print(f'**Transformation after column {k+1}:\n')
        # Extract each column from input matrix as a vector to wokr on
        current_column = result_matrix[k:, k].astype(float) 
        e = H_matrix[k:, k] # Get e from identity matrix for calculation of vector
        if current_column[0] > 0: # Check for sign of a for calculation of alpha
            alpha = -np.linalg.norm(current_column,2)
        else:
            alpha = np.linalg.norm(current_column,2)

        # Calculate vector using current_column vector transpose, alpha, and e
        vector[k:, k] = np.around(result_matrix[k:, k].T - alpha * e, decimals=0)
        # Calculate beta using vector transpose and vector
        beta = np.around(vector[k:, k].T @ vector[k:, k], decimals=0)
        if beta == 0: # Avoid division by 0 when transforming the remaining submatrix
            continue
        
        print(f'alpha{k+1} = {alpha}, e{k+1} = {e}, beta{k+1} = {beta}\n')
        print(f'vector{k+1} = {vector[k:, k]}\n')


        return result_matrix

M = np.array([[1, -1, 4],
            [1, 4,-2],
            [1, 4, 2],
            [1, -1, 0]], dtype=float)
result_matrix = householder(M) 