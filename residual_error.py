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
        print('Error: Vector contains more than 1 column or has mismatched number of elements') 
        return
    
    # Initializations
    n = A_matrix.shape[0] # Get number of rows
    i = 0 # Row iteration
    I_matrix = np.identity(n, dtype=float)
    
    # Gauss elimination phase
    while i < n:
        if A_matrix[i][i] == 0:
            print('Error: The pivoting cannot be zero')
            return
        # Iterate from the second row to n row
        for j in range(i+1, n):
            scaling = A_matrix[j][i] / A_matrix[i][i] # Current element divided by pivoting
            I_matrix[j][i] = scaling # Produce Lower triangular system of n x n matrix
            A_matrix[j] = A_matrix[j] - scaling * A_matrix[i] # Produce Upper triangular system of n x n matrix
        i += 1

    # Solve the given system
    return backward_substitution(n, A_matrix, forward_substitution(n, I_matrix, b))


def forward_substitution(n, L_matrix, b):
    # Initialize vector y
    y = np.zeros(n) 
    for i in range(n):
        y[i] = ((b[i] - np.dot(L_matrix[i, :i], y[:i])) / L_matrix[i, i])
    return y


def backward_substitution(n, U_matrix, y):
    # Initialize vector x
    x = np.zeros(n) 
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U_matrix[i, i+1:], x[i+1:])) / U_matrix[i, i]
    return x


def generatorHb(n):
    H_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            H_matrix[i, j] = 1 / (i + j + 1)
    return H_matrix


def residual_calculation(H_matrix, b, x_hat):
    residual=[]
    Hx = np.dot(H_matrix, x_hat)
     
    for  i in range(len(b)):
        residual_val = b[i] - Hx[i]
        print(f'i={i}\n')
        print(f'b[i]={ b[i]: .30f} and dataype = {type(b[i])}\n')
        print(f'Hx[i]={ Hx[i]: .30f} and dataype = {type(Hx[i])}\n')
        print(f'VALUE = { residual_val}\n')
        residual.append(residual_val)
    #residual = b - np.dot(H_matrix, x_hat)
    # Calcualte residual infinite norm
    res_inf_norm = np.linalg.norm(residual, np.inf)
    return residual, res_inf_norm

def error_norm_calculation(x_true, x_hat):
    error = x_hat - x_true # x_delta = x_hat - x_true
    error_inf_norm = np.linalg.norm(error, np.inf) # Get error infinite norm
    x_true_norm = np.linalg.norm(x_true, np.inf) # Get true x infinite norm
    error_percentage = (error_inf_norm / x_true_norm)
    return error, error_percentage

def condH_calculation(H_matrix):
    condH = np.linalg.cond(H_matrix, np.inf)
    return condH

n = 2  # Initialize n with 2
res_normList = []
condH_list = []
while True:
    # Generate Hilbert matrix, true solution vector x, and b
    H_matrix = generatorHb(n)
    x_true = np.transpose(np.ones(n))  # n-vector with all entries equal to 1
    b = np.dot(H_matrix, x_true)  # Compute b = Hx
    
    # Use Gaussian elimination to get x hat, calculate residual, error and condition H
    x_hat = gauss_elimination_algo(H_matrix.copy(), b.copy())
    #print(f'for n = {n} x_hat ={ x_hat} and dataype = {type(x_hat)}\n')

    residual, res_inf_norm = residual_calculation(H_matrix, b, x_hat)
    error, error_percentage = error_norm_calculation(x_true, x_hat)
    condH = condH_calculation(H_matrix)

    # Get list of infinite norms and condH to print out
    res_normList.append(float(res_inf_norm))
    condH_list.append(condH)
    #print(f'For n = {n}, we have:\nx hat = {x_hat}\nx delta = {error}\nr = {residual}\nerror percentage = {error_percentage:.4%}\n')
    
    # Exit loop if the error percentage has passed 100%
    if error_percentage > 1:
        break  
    n += 1

# print('--------------------------------------------------\n') 
# print(f'n\tr infinite norm\t\t\tCond(H)')
# n_loop = 2
# for i in range(len(res_normList)):
#     print(f'{n_loop}\t{res_normList[i]:.5e}\t\t\t{condH_list[i]:.3f}')
#     n_loop += 1

