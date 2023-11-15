import numpy as np

def X_Y(n):
    global a, r, p
    # Form the f(x) equation (with x = n years) from the loan payment values 
    f_n = a*(1+r)**n-p*(((1+r)**n-1)/r)
    return f_n

def X_Y_derivative(n, h):
    # Calculate f'(x) using h as the change in n years in each iteration 
    f_nprime = (X_Y(n+h) - X_Y(n))/h
    return f_nprime

def NewtonA(n_guess):
    global h0, a, r, p
    k = 0
    while True:
        if X_Y_derivative(n_guess, h0) == 0:
            print('Calculation error, please use a different guess for n\n')
            break
        h = -X_Y(n_guess)/X_Y_derivative(n_guess, h0) # h is the change in n years in each iteratio
        n_next = n_guess + h # Calculate the next guessed years
        print(f'{k}\t\t {n_guess:.4f}\t\t\t{X_Y(n_guess):.4f}')

        # Check for error
        error = abs(n_next - n_guess)/n_next
        if error <= h0:
            break
        k+=1
        n_guess=n_next

    return n_guess, int(X_Y(n_guess))

a = 100000
r = 0.06
p = 10000
h0 = 1e-6
n_initial = 10
print(f'\nInitial guess is n = {n_initial} years\n')
print(f'k \t\t n\t\t\t\t X-Y')
n, x_y = NewtonA(n_initial)
print(f'\nIt will need approximately {np.around(n,decimals=0)} years to repay the loan, when X-Y={x_y}')
