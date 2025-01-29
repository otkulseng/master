from bodge import Hamiltonian, CubicLattice, sigma0, sigma3, jsigma2
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import csr_matrix, coo_matrix
from chebyshev import cheby_from_dct
import time

from scipy.sparse.linalg import cg

from symbolic import get_sparse_newman_functionals

from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import connected_components



def rel_diff(a, b, eps=1e-10):
    return np.abs(a - b) / (eps + np.abs( a + b))


def gram_schmidt(A):

    m, n = A.shape
    Q = np.zeros((m, n))

    for j in range(n):
        q = A[:, j]

        for i in range(j):
            q -= np.dot(Q[:, i], A[:, j]) * Q[:, i]

        Q[:, j] = q / np.linalg.norm(q)

    return Q

def stochastic_trace_estimation(matr: csr_matrix, chebyshev_coefficients: np.ndarray, num_vectors: int = None):
    # Maybe not generate all at the same time if n is huge, or maybe do?
    if num_vectors is None:
        num_vectors = int(np.log(matr.shape[0]))
        num_vectors = int(np.sqrt(matr.shape[0]))
    # rand_vecs = generate_random_orthogonal_vectors(matr.shape[0], num_vectors)

    rand_vecs = 2 * (np.random.randint(0, 2, size=(matr.shape[0], num_vectors))) - 1

    # Krylov probes
    # rand_vecs = np.zeros((matr.shape[0], num_vectors))
    # rand_vecs[:, 0] = np.random.randn(matr.shape[0])
    # for i in range(1, num_vectors):
    #     rand_vecs[:, i] = matr.dot(rand_vecs[ :, i-1])



    # rand_vecs,_  = np.linalg.qr(rand_vecs)
    # rand_vecs = gram_schmidt(rand_vecs)

    # meanvec = np.mean(rand_vecs, axis=1)
    # rand_vecs = rand_vecs - meanvec[:, np.newaxis]

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

def get_trace(Q, P, v):
    # Apply numerator
    numerator = P(v)

    # Apply denominator
    res, _ = cg(Q, numerator)

    return np.dot(v, res)



def stochastic_newman_trace_estimation(matr: csr_matrix, num_vectors: int = 10, order: int  = 10):
    Q, P = get_sparse_newman_functionals(order, matr)

    rand_vecs = np.random.randn(num_vectors, matr.shape[0]).astype(np.complex128)

    traces = [
        get_trace(Q, P, v) for v in rand_vecs
    ]

    return np.real(np.mean(traces))


def make_blocks(matr: csr_matrix) -> list[csr_matrix]:
    num_blocks, labels = connected_components(csgraph=(matr != 0).astype(int))

    perm = np.argsort(labels)

    A_perm = matr[perm, :][:, perm]

    block_size = matr.shape[0] // num_blocks

    blocks = [
        A_perm[block_size*i: block_size*(i+1), :][:, block_size*i: block_size*(i+1)] for i in range(num_blocks)
    ]

    return blocks


class NewHamiltonian(Hamiltonian):
    def __init__(self, lattice):
        super().__init__(lattice)

    def free_energy(self, temperature = 0, cuda = False):
        assert(temperature >= 0)
        temperature = temperature


        def stochastic_trace(M: csr_matrix):
            # Need to normalize the matrix. Choose inf norm as a compromise between
            # efficiency and smallness (calculating the spectral norm not feasible)
            norm = sparse_norm(M, ord=1)

            # Scale matrix to ensure eigenvalues are in the interval [-1, 1]
            M = M / norm

            def free_energy_func(x):
                abs_term = norm * np.abs(x) / 2
                if temperature > 0:
                    abs_term += temperature * np.log(1 + np.exp(- norm / temperature * np.abs(x)))
                return abs_term
                # Above equivalent to T ln(2cosh(norm x / 2T))
                # return - temperature * np.log(2) + norm * np.abs(x)  + temperature * np.log(1 + np.exp(- A * np.abs(x)))


            # TODO: Remove magic number
            chebyshev_coefficients = cheby_from_dct(free_energy_func, 50)

            first = chebyshev_coefficients[0]
            chebyshev_coefficients = chebyshev_coefficients / first

            # Using the chebyshev coefficients, evaluate the trace stochastically
            trace = stochastic_trace_estimation(M, chebyshev_coefficients)

            # calculate the free energy
            # F = - T / 2 * Tr(ln(2cosh(H/2T)))
            return - first *  trace / 2

        # Convert to csr matrix for efficient computation
        matrix = self.matrix('csr')

        # Permute the matrix into blocks
        blocks = make_blocks(matrix)

        # Use stochastic trace estimation on each block and sum the results
        # return stochastic_trace(matrix)
        return np.sum(
            [stochastic_trace(blk) for blk in tqdm(blocks)]
        )


    def newman(self, temperature = 0, order=10):
        # Use the newman https://arxiv.org/pdf/2005.02736
        # approximation for |x|

        # Convert to csr matrix for efficient computation
        matrix = self.matrix('csr')

        # Get the largest singular value
        norm = sparse_norm(matrix, ord=1)

        # Scale matrix to ensure eigenvalues are in the interval [-1, 1]
        matrix = matrix / norm

        return norm * stochastic_newman_trace_estimation(
            matrix, num_vectors=50, order=10
        ) / 2

    def fourier(self, axis):

        matr = self.matrix('csr')

        blocks = make_blocks(matr)



def testing():
    times = []

    Lx = 1000
    Ly = 10
    Lz = 10

    lattice = CubicLattice((Lx, Ly, Lz))
    system = NewHamiltonian(lattice)
    t = 1
    mu = -4 * t
    m = 0.05 * t
    DS = 0.10 * t
    T = 1e-4

    with system as (H, D):
        for i in lattice.sites():
            H[i, i] = -mu * sigma0 -m * sigma3
            D[i, i] = -DS * jsigma2

        for i, j in lattice.bonds():
            H[i, j] = -t * sigma0

        # for i, j in lattice.edges():
        #     H[i, j] = -t * sigma0



    en = system.free_energy(temperature=0)
    print(en)
    # matr = system.matrix()



    # plt.imshow(np.abs(matr), cmap='hot', interpolation='nearest')
    # plt.colorbar()  # Add a color scale

    # blocks = system.fourier(0)
    # for block in blocks:
    #     plt.figure()
    #     plt.imshow(np.abs(block.todense()), cmap='hot', interpolation='nearest')
    #     plt.colorbar()  # Add a color scale
    # plt.show()


    # print(kx)
    # plt.plot(kx)
    # plt.show()

    # eigvals, _ = system.diagonalize()
    # print(f"Eigendecomp: {time.time() - mid}")
    # print(-np.sum(eigvals))








if __name__ == '__main__':
    testing()