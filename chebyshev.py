import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval


def func(x, beta=0.01, eps=1e-4):
    return 1 + np.exp(-beta * np.sqrt(x**2 + eps**2))



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
    cvec[1::2] = 0

    return cvec[np.nonzero(cvec)]

def reconstruct_even(x, coeffs):
    # Reconstructs the chebyshev coefficients under the assumption
    # that all odd powers are zero

    res = np.zeros((x.size, coeffs.size))
    res[:, 0] = np.ones_like(x)
    res[:, 1] = 2 * x**2 - 1
    for i in range(2, coeffs.size):
        res[:, i] = 2 * (2*x**2 - 1) * res[:, i-1] - res[:, i-2]

    return np.einsum("ij, j->i", res, coeffs)

def rel_diff(a, b, eps=1e-10):
    return np.abs(a - b) / (eps + np.abs( a + b))
def main():
    x = np.linspace(-1, 1, 100)

    N = 100
    c = cheby_from_dct(func, N)

    y_approx = reconstruct_even(x, c)
    print(c)

    plt.plot(x,np.log10( rel_diff(func(x, eps=0), y_approx)), label='relative diff')
    # plt.plot(x, func(x, eps=0), label=' exact')
    # plt.plot(x, y_approx, label='approx', marker="+")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
