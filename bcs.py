from scipy.fft import dctn, idctn, dstn, idstn
import numpy as np
import matplotlib.pyplot as plt


def multidimensional_cosine_expansion(coefficients: np.ndarray):
    # Expand in each dimension
    return idctn(coefficients, type=2, norm='ortho')

def multidimensional_fast_cosine_transform(signal: np.ndarray):
    return dctn(signal, type=2,norm='ortho')

def main():

    N = 100

    orig = np.linspace(-1, 1, N)
    c = multidimensional_fast_cosine_transform(orig)
    # c[np.abs(c) < 1e-5 * np.max(np.abs(c))] = 0

    print(c[np.nonzero(c)].size)
    print(c)

    x = multidimensional_cosine_expansion(c)
    plt.plot(x)

    # plt.savefig('main.png')

    plt.show()
if __name__ == '__main__':
    main()






