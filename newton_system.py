import numpy as np

def f(x_guesses):
    x1 = x_guesses[0]
    x2 = x_guesses[1]
    # Define system of non-linear equations
    equation1 = x1 + 2 * x2 - 2
    equation2 = x1**2 + 4 * x2**2 - 4
    return np.array([equation1, equation2])

def jacobian(x_guesses):
    global h0
    n = len(x_guesses)
    jacobian_matrix = np.zeros((n,n))

    for j in range (n):
        new_guesses = x_guesses.copy().astype(float)
        new_guesses[j] += h0
        # Calculate partial derivative from the values pair
        f_prime = (f(new_guesses) - f(x_guesses)) / h0
        jacobian_matrix[:, j] = f_prime
    return jacobian_matrix


def NewtonB(x_guesses):
    global h0
    print(f'k \t | x1_k \t | x2_k')
    k = 0
    while True:
        s_k = np.linalg.inv(jacobian(x_guesses)) @ (-1 * f(x_guesses))
        x_k1 = x_guesses + s_k

        # Check for error
        error = np.linalg.norm((x_k1-x_guesses) / x_k1)
        if error <= h0:
            break
        # Update x guesses
        x_guesses = x_k1
        k += 1
        print(f'{k} \t | {x_guesses[0]:.6f} \t | {x_guesses[1]:.6f}')
    return x_guesses

h0 = 1e-4
initial_guesses = np.array([1,1])
print(f'Take initial guess for x1 = 1, x2 = 1:')
root_results = NewtonB(initial_guesses)