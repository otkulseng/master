import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct


def gen_func(beta):
    def func(x):
        return np.log(2 * np.cosh(beta * x / 2))
    return func



# def gen_func(beta, eps):
#     def func(x):
#         return 1 + np.exp(-beta * x)
#     return func

# def gen_funcg(beta, eps):
#     def func(x):
#         return  np.sqrt(x**2 + eps**2)
#     return func

def cheby_from_dct(f, N):

    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    fx = f(x)

    # Use the dct (discrete cosine transform) to get
    # the chebyshev nodes.
    c_fourier = dct(fx, type=1)
    cvec = np.zeros(N + 1, like=c_fourier)
    cvec = c_fourier / N
    cvec[0] /= 2
    cvec[N] /= 2

    # # cvec[1::2] = 0
    # return cvec[::2] # Keep only even nodes
    cvec[1::2] = 0
    return cvec[np.nonzero(cvec)] # Keep only even nodes

def reconstruct_even(x, coeffs):
    # Reconstructs the chebyshev coefficients under the assumption
    # that all odd powers are zero


    res = np.zeros((x.size, coeffs.size))
    res[:, 0] = np.ones_like(x)
    if coeffs.size > 1:
        res[:, 1] = 2 * x**2 - 1
    for i in range(2, coeffs.size):
        res[:, i] = 2 * (2*x**2 - 1) * res[:, i-1] - res[:, i-2]

    return np.einsum("ij, j->i", res, coeffs)

def rel_diff(a, b, eps=1e-10):
    return np.abs(a - b) / (eps + np.abs( a + b))


def chebyshev_accuracy_plot():
    betas = np.linspace(1e-10, 1000, 100)
    threshold = 1e-4

    x = np.linspace(-1, 1, 101)
    res = []

    for beta in betas:
        func = gen_func(beta)
        fvals = func(x)
        low = 2
        high = 1000
        while high > low + 1:
            N = (high + low) // 2
            c = cheby_from_dct(func, N)

            y_approx = reconstruct_even(x, c)
            cur = np.max(rel_diff(y_approx, fvals))
            if cur < threshold:
                high = N
            else:
                low = N

            print(low, high, cur)
        res.append(N)

    plt.plot(betas, res)
    plt.show()

def main():
    chebyshev_accuracy_plot()



def func(x):
    return np.log(2*np.cosh(x/2))

def main():

    x = np.linspace(-1e-4, 1e-4, 100)

    N = 200
    c = cheby_from_dct(func, N)

    y_approx = reconstruct_even(x, c)
    print(c)

    plt.plot(x,np.log10( rel_diff(func(x), y_approx)), label='relative diff')
    plt.figure()
    plt.plot(x, func(x), label="Exact")
    plt.plot(x, y_approx, label="Approxyt")

    # plt.xlim([-1e-4, 1e-4])
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
