from bodge import *
from scipy.sparse.csgraph import connected_components
import numpy as np
import matplotlib.pyplot as plt

def make_blocks(matr):
    num_blocks, labels = connected_components(csgraph=(matr != 0).astype(int))

    perm = np.argsort(labels)
    print(matr)
    A_perm = matr[perm, :][:, perm]
    print(A_perm)

    print(num_blocks)

    plt.imshow(A_perm)
    plt.colorbar()
    plt.show()

    block_size = matr.shape[0] // num_blocks

    blocks = [
        A_perm[block_size*i: block_size*(i+1), :][:, block_size*i: block_size*(i+1)] for i in range(num_blocks)
    ]

    vals = [
        np.linalg.eig(blk)[0] for blk in blocks
    ]
    print(vals)



lattice = CubicLattice((3, 1, 1))
system = Hamiltonian(lattice)

t = 1
μ = -3 * t
Δs = 2*  t
m = 4 * t

with system as (H, Δ):
    for i in lattice.sites():
        H[i, i] = -μ * σ0 #+ m * sigma1
        # Δ[i, i] = -Δs * jσ2
    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

matr = system.matrix().astype(int)
make_blocks(matr)