import numpy as np


# Create lambda functions to calculate f(x) and f'(x)
f = lambda x: (x - 2)**2 + 4*x - 8
f_prime = lambda x,h: (f(x+h)-f(x))/h

def newtonA(x_range1, x_range2):
    h0 = 1e-5
    for x in range (x_range1+1,x_range2+1):
        print(f'\nInitial guess x = {x}')
        print(f"k \t\t x_k\t\t\t f(x_k) \t\tf'(x_k) \t\th_k")
        k = 0
        while True:
            if f_prime(x, h0) == 0:
                print("Error: f'(x) = 0")
                break
            # Apply newton method
            h = -f(x)/f_prime(x,h0)
            x_k1 = x + h
            print(f'{k}\t\t {x:.6f}\t\t{f(x):.6f}\t\t{f_prime(x,h0):.6f}\t\t{(h):.6f}')

            # Check for error against convergence tolerance
            error = abs(x_k1-x)/x_k1
            if error <= h0:
                break
            k += 1
            x = x_k1

    return x

x_root = newtonA(0,4)



