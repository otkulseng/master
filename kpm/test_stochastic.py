from chebyshev import cheby_from_dct
from stochastic_trace_estimation import *
from bodge import *
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm


def rel_change(a, b, tol=1e-15):
    return np.abs(a-b) / (np.abs(a + b) + tol)


def test_stochastic():

    N = 1000
    num_vec = 100
    lat = CubicLattice((20, 20, 1))



    ham = Hamiltonian(lat)
    with ham as (H, D):
        for i in lat.sites():
            H[i, i] = -1.0 * sigma0  + 0.4 * sigma2
            D[i, i] = -0.5 * jsigma2
        for i, j in lat.bonds():
            H[i, j] = -1.0 * sigma0

    norm = sparse_norm(ham.matrix('bsr'))

    def func(x):
        return np.abs(norm * x)

    coefs = cheby_from_dct(func, N)

    vecs = (torch.randint(0, 2, (num_vec, 4 * lat.size)) * 2 - 1).to(torch.complex128)

    stoch = chebyshev_stochastic_estimation(
        LinearOperator(torch.tensor(ham.matrix() / norm)),
        coefs,
        vecs
    ).detach().numpy()


    truth = np.sum(func(np.linalg.eigvals(ham.matrix()) )/norm)
    print(truth, stoch, rel_change(truth, stoch))
    assert rel_change(truth, stoch) < 1e-2

def test_clenshaw():

    N = 1000
    num_vec = 100
    lat = CubicLattice((20, 20, 1))
    ham = Hamiltonian(lat)
    with ham as (H, D):
        for i in lat.sites():
            H[i, i] = -1.0 * sigma0  + 0.4 * sigma2
            D[i, i] = -0.5 * jsigma2
        for i, j in lat.bonds():
            H[i, j] = -1.0 * sigma0

    norm = sparse_norm(ham.matrix('bsr'))

    def func(x):
        return np.abs(norm * x)

    coefs = cheby_from_dct(func, N)

    vecs = (torch.randint(0, 2, (num_vec, 4 * lat.size)) * 2 - 1).to(torch.complex128)

    stoch = clenshaw_chebyshev_stochastic_estimation(
        LinearOperator(torch.tensor(ham.matrix() / norm)),
        coefs,
        vecs
    ).detach().numpy()


    truth = np.sum(func(np.linalg.eigvals(ham.matrix()) )/norm)
    print(truth, stoch, rel_change(truth, stoch))

    assert rel_change(truth, stoch) < 1e-2

