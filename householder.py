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


M = np.array([[1, -1, 4],
            [1, 4,-2],
            [1, 4, 2],
            [1, -1, 0]], dtype=float)
result_matrix = householder(M) 