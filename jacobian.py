import numpy as np

def equation_calc(values_pair):
    global gamma, delta
    x = values_pair[0]
    y = values_pair[1]
    # Calculate the 2 equations according to x and y
    equation1 = gamma * x * y - x * (1 + y)
    equation2 = -x * y + (delta - y) * (1 + y)
    return np.array([equation1, equation2])

def jacobian_matrix(values_pair):
    global h0
    n = len(values_pair)
    jacobian_matrix = np.zeros((n,n))

    for j in range (n):
        new_values = values_pair.copy().astype(float)
        new_values[j] += h0
        # Calculate partial derivative from the values pair
        derivative = (equation_calc(new_values) - equation_calc(values_pair)) / h0
        jacobian_matrix[:, j] = derivative
    return jacobian_matrix


def NewtonB(values_pair):
    global h0
    print(f'k \t | x \t\t | y \t\t | equation 1\t | equation 2')
    k = 0
    while True:
        equation_values = equation_calc(values_pair)
        s_k = np.linalg.inv(jacobian_matrix(values_pair)) @ (-1 * equation_calc(values_pair))
        values_pair_1 = values_pair + s_k

        # Check for error
        error = np.linalg.norm((values_pair_1-values_pair) / values_pair_1)
        if error <= h0:
            break
        # Update value pairs
        values_pair = values_pair_1
        k += 1
        print(f'{k} \t | {values_pair[0]:.5f} \t | {values_pair[1]:.5f} \t | {equation_values[0]:.2f} \t | {equation_values[1]:.2f}')
    print(f'\nFinal result:\nx = {np.around(values_pair[0],decimals=2)}, y = {np.around(values_pair[1],decimals=2)}')
    return values_pair


gamma = 5
delta = 1
h0 = 1e-6
values_pair = np.array([7,0.5])
list_of_values = NewtonB(values_pair)