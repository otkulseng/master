import numpy as np
from kpm.chebyshev import cheby_from_dct, rel_diff
from scipy.sparse import csr_matrix
from newbodge import *
import matplotlib.pyplot as plt

def approximate_largest_eigenvalue(matr: csr_matrix, rel_tol=1e-6, max_iter=1000):
    # Assumes matr is symmetric
    v0 = np.random.randn(matr.shape[0])
    v0 = v0 / np.linalg.norm(v0)
    elems = []
    for _ in range(max_iter):
        v1 = matr.dot(v0)
        elems.append(np.linalg.norm(v1))
        v0 = v1 / elems[-1]

        if len(elems) < 2:
            continue

        if rel_diff(elems[-1], elems[-2]) < rel_tol:
            return elems[-1] * (1 + np.sqrt(rel_tol))

        elems = [elems[-1]]


    raise RuntimeError("Could not find largest eigenvalue")






def testing():
    lattice = CubicLattice((100, 100, 1))
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


    # eigvals = np.real(np.linalg.eigvals(system.matrix()))
    # print("Eigvals done")

    # true_val = np.sum(np.abs(eigvals))
    true_val = 1


    sparse_matr = system.matrix(format="csr")
    # largest = approximate_largest_eigenvalue(sparse_matr)

    # ratio = np.max(np.abs(eigvals)) / largest
    # print(ratio)
    # assert(ratio < 1)




    approx = trace_of_absolute_value(sparse_matr, 100, 100)

    print(true_val, approx, rel_diff(true_val, approx))

def main():
    testing()
if __name__ == '__main__':
    main()
