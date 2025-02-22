from chebyshev import *
import numpy as np
import matplotlib.pyplot as plt

def test_compose():
    def f(x, eps=1e-1):

        return np.sqrt(x**2 + eps**2)

    x = np.linspace(-1, 1, 100)


    coefs = cheby_from_dct(f, 200, even=True)
    print(coefs)

    plt.plot(x, f(x))
    plt.savefig("both.pdf")


test_compose()