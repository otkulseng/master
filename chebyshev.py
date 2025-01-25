import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval

def func(x, beta=8.0):
    return 1 + np.exp(-beta * x)

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
    return cvec



def main():
    x = np.linspace(-1, 1, 100)

    c = cheby_from_dct(func, 100)
    y_approx = chebval(x, c)
    print(c)

    plt.plot(x, func(x), label='exact')
    plt.plot(x, y_approx, label='approx')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
