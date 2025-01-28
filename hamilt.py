from bodge import Hamiltonian, CubicLattice, sigma0, sigma3, jsigma2
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import csr_matrix
from chebyshev import cheby_from_dct

import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh





def rel_diff(a, b, eps=1e-10):
    return np.abs(a - b) / (eps + np.abs( a + b))


def stochastic_trace_estimation(matr: csr_matrix, chebyshev_coefficients: np.ndarray, num_vectors: int = None):
    # Maybe not generate all at the same time if n is huge, or maybe do?
    # rand_vecs = generate_random_orthogonal_vectors(matr.shape[0], num_vectors)
    if num_vectors is None:
        num_vectors = int(np.sqrt(matr.shape[0]))

    rand_vecs = np.random.randn(matr.shape[0], num_vectors)
    meanvec = np.mean(rand_vecs, axis=1)
    rand_vecs = rand_vecs - meanvec[:, np.newaxis]

    # N x num_vectors
    # First iteration, store all vectors
    # Chebyshev order x matrix_size x num_vectors
    tmatr = np.zeros((chebyshev_coefficients.size, rand_vecs.shape[0], rand_vecs.shape[1]), dtype=matr.dtype)
    tmatr[0, :, :] = rand_vecs
    tmatr[1, :, :] = 2 * matr.dot(matr.dot(rand_vecs)) - rand_vecs


    for n in range(2, chebyshev_coefficients.size):
        # Calculate next term
        tmatr[n, :, :] = 2 * (2 * matr.dot(matr.dot(tmatr[n-1, :, :])) - tmatr[n-1, :, :]) - tmatr[n-2, :, :]


    return np.real(np.einsum(
        "n, ij, nij->", chebyshev_coefficients, rand_vecs, tmatr
    )) / num_vectors

class NewHamiltonian(Hamiltonian):
    def __init__(self, lattice):
        super().__init__(lattice)

    def free_energy(self, temperature = 0, cuda = False):
        assert(temperature >= 0)
        # TODO: Handle temperature==0 better.
        temperature = 1e-5 + temperature
        print(temperature)
        # Convert to csr matrix for efficient computation
        matrix = self.matrix('csr')

        # Get the largest singular value
        norm = sparse_norm(matrix, ord=1)

        # Scale matrix to ensure eigenvalues are in the interval [-1, 1]
        matrix = matrix / norm

        # The function f : [-1, 1] -> R which is going to be chebyshev approximated
        print(norm, temperature, norm / temperature)

        def free_energy_func(x):
            return temperature * np.log(2*np.cosh(norm / temperature * x / 2))

        # TODO: Remove magic number
        chebyshev_coefficients = cheby_from_dct(free_energy_func, 50)
        print(chebyshev_coefficients)

        # Using the chebyshev coefficients, evaluate the trace stochastically
        trace = stochastic_trace_estimation(matrix, chebyshev_coefficients)

        # calculate the free energy
        # F = - T / 2 * Tr(ln(2cosh(H/2T)))
        return - trace / 2



def testing():
    lattice = CubicLattice((10, 10, 1))
    system = NewHamiltonian(lattice)

    t = 1
    mu = -3 * t
    m = 0.05 * t
    DS = 0.10 * t

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -mu * sigma0 -m * sigma3
            Δ[i, i] = -DS * jsigma2
        for i, j in lattice.bonds():
            H[i, j] = -t * sigma0

    T = 0
    print(system.free_energy(T))







if __name__ == '__main__':
    testing()