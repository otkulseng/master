from bodge import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chebyshev import cheby_from_dct, reconstruct_even, rel_diff

def func(x, eps=1e-4):
    return np.sqrt(x**2 + eps**2)


def absval_func(x):
    return np.abs(x)

def approx_absolute_value(x, N):
    c = cheby_from_dct(func, N)
    return reconstruct_even(x, c)


def reconstruct_even_matrix(x, coeffs):
    # Reconstructs the chebyshev coefficients under the assumption
    # that all odd powers are zero

    res = np.zeros((coeffs.size, x.shape[0], x.shape[1]))
    res[0, :, :] = np.eye(x.shape[0])

    if coeffs.size > 1:
        res[1, :, :] = 2 * x @ x - res[0, :, :]
    for i in range(2, coeffs.size):
        res[i, :, :] = 2 * (2*x@ x - res[0, :, :]) @ res[i-1, :, :] - res[i-2, :,:]

    return np.einsum("kij, k->ij", res, coeffs)

def approx_absolute_trace(matr, N):
    coefs = cheby_from_dct(np.abs, N)
    return np.trace(
        reconstruct_even_matrix(matr, coefs)
    ) / 2



def testing():
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    t = 1
    μ = -3 * t
    m = 0.05 * t
    Δs = 0.10 * t

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 -m * σ3
            Δ[i, i] = -Δs * jσ2
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0


    matr = system.matrix()
    matr = matr / np.linalg.norm(matr)

    eigvals = np.linalg.eigvals(matr)
    eigvals = np.real(eigvals[eigvals > 0])

    plt.plot(eigvals)
    plt.show()

    temp = 0.0

    en = system.free_energy(temp)
    print(en)
    true_val = np.sum(eigvals)

    Nvals = np.arange(20)*5 + 100
    tests = np.array([
        rel_diff(approx_absolute_trace(matr, n), true_val) for n in tqdm(Nvals)
    ])
    print(tests)
    plt.plot(Nvals, np.log10(tests))
    plt.show()
    # eigvals, _ = system.diagonalize(format='raw')


    # plt.plot(eigvals)
    # plt.show()
    # print(matr)

if __name__ == '__main__':
    testing()