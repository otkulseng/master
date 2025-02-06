
from bodge import *
from compress import *
import torch

from scipy.sparse.linalg import norm as sparse_norm
# Test of BlockSparseMatrix

def test_base_matrix():
    lat = CubicLattice((10, 10, 10))
    num_vec = 100

    ham = PotentialHamiltonian(lat)
    bodge_ham = Hamiltonian(lat)


    with ham as (H, V):
        for i in lat.sites():
            x, y, z = i
            H[i, i] = - 1.0 * sigma0 + 0.4 * sigma2

        for i, j in lat.bonds():
            H[i, j] = -1.0 * sigma0


    with bodge_ham as (H, _):
        for i in lat.sites():
            H[i, i] = -1.0 * sigma0  + 0.4 * sigma2
        for i, j in lat.bonds():
            H[i, j] = -1.0 * sigma0

    shape = (num_vec, 2 * lat.size, 2)
    vecs = torch.randn(shape)


    block_sparse_matrix = ham.base_matrix()
    res1 = block_sparse_matrix.matvec(vecs.to(torch.complex128)).detach().numpy().reshape((num_vec, -1))

    res2 = np.einsum("ij, nj -> ni", bodge_ham.matrix(), vecs.detach().numpy().reshape((num_vec, -1)))

    print(np.max(np.abs(res1.flatten() - res2.flatten())))
    assert np.allclose(res1, res2, atol=1e-4)


def test_order_parameters():
    Lx, Ly, Lz = (10, 10, 10)
    lat = CubicLattice((Lx, Ly, Lz))
    num_vec = 100

    ham = PotentialHamiltonian(lat)
    bodge_ham = Hamiltonian(lat)

    order_param = (np.arange(lat.size) + 1)/ lat.size


    with ham as (_, V):
        for i in lat.sites():
            V[i, i] = 1


    with bodge_ham as (_, D):
        for i in lat.sites():

            D[i, i] = -order_param[lat.index(i)] * jsigma2

    shape = (num_vec, 2 * lat.size, 2)
    vecs = torch.randn(shape).to(torch.complex128)

    indices, _ = ham.order_matrix()

    order_param = torch.tensor(order_param).to(torch.complex128)
    res1 = apply_order_parameters(indices, torch.concatenate([order_param, -order_param]), vecs).detach().numpy().reshape((num_vec, -1))
    res2 = np.einsum("ij, nj -> ni", bodge_ham.matrix(), vecs.detach().numpy().reshape((num_vec, -1)))


    diff = np.abs(res1.flatten() - res2.flatten())
    print(np.mean(diff), np.std(diff), np.max(diff), np.min(diff))
    assert np.allclose(res1, res2, atol=1e-4)
