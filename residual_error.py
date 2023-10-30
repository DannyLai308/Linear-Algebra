# Student Name: Danny Lai - 400491405
# Class: Linear Algebra - 4MA3
import numpy as np

def forward_substitution(A_matrix, b_vector, n):
    L_matrix = np.identity(n, dtype=float)
    augmented_matrix = np.concatenate((A_matrix, np.reshape(b_vector, (n,1))), axis=1)
    for j in range(n-1):
        pivot = augmented_matrix[j, j]
        if pivot == 0:
            continue
        for i in range (j+1, n):
            scaling = augmented_matrix[i,j] / pivot
            augmented_matrix[i, j:] = augmented_matrix[i, j:] - scaling * augmented_matrix[j, j:] # Produce Upper matrix
            L_matrix[i,j] = scaling # Produce lower matrix
    return augmented_matrix

def backward_substitution(A_matrix, b_vector, n):
    augmented_matrix = forward_substitution(A_matrix, b_vector, n)
    x=np.zeros(n)
    for j in range(n-1, -1, -1):
        pivot = augmented_matrix[j,j]
        if pivot == 0:
            continue
        x[j] = augmented_matrix[j, -1]/pivot
        for i in range(j-1,-1,-1):
            augmented_matrix[i, -1] = augmented_matrix[i, -1] - augmented_matrix[i, j] * x[j]
    return x

def gauss_elimination_algo (A_matrix, b_vector, n):
    augmented_matrix = forward_substitution(A_matrix, b_vector, n)
    x = backward_substitution(A_matrix, b_vector, n)
    return augmented_matrix, x

def generatorHb(n):
    H_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            H_matrix[i, j] = 1 / (i + j + 1)
    return H_matrix

def residual_calculation(H_matrix, b, x_hat):
    # Calculate residual 
    residual = b - np.dot(H_matrix, x_hat)
    # Calcualte residual infinite norm
    res_inf_norm = np.linalg.norm(residual, np.inf)
    return residual, res_inf_norm

def error_norm_calculation(x_true, x_hat):
    error = x_hat - x_true # x_delta = x_hat - x_true
    error_inf_norm = np.linalg.norm(error, np.inf) # Get error infinite norm
    x_true_norm = np.linalg.norm(x_true, np.inf) # Get true x infinite norm
    error_percentage = (error_inf_norm / x_true_norm) # Get error percentage
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
    U_matrix, x_hat = gauss_elimination_algo(H_matrix, b, n)
    residual, res_inf_norm = residual_calculation(H_matrix, b, x_hat)
    error, error_percentage = error_norm_calculation(x_true, x_hat)
    condH = condH_calculation(H_matrix)

    # Get list of infinite norms and condH to print out
    res_normList.append(float(res_inf_norm))
    condH_list.append(condH)
    print(f'For n = {n}, we have:\nx hat = {x_hat}\n\nx delta = {error}\n\nr = {residual}\n\nerror percentage = {error_percentage:.4%}\n')
    
    # Exit loop if the error percentage has passed 100%
    if error_percentage > 1:
        break  
    n += 1

print('--------------------------------------------------\n') 
print(f'n\tr infinite norm\t\t\tCond(H)')
n_loop = 2
for i in range(len(res_normList)):
    print(f'{n_loop}\t{res_normList[i]:.5e}\t\t\t{condH_list[i]:.3f}')
    n_loop += 1
print(f'\n At n = {n_loop-1}, the error goes pass 100%\n')
