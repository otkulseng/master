from bodge import *
from tqdm import tqdm
import torch
from scipy.sparse.csgraph import connected_components
import numpy as np


class BlockReuseMatrix:
    # Maybe faster if not dictionary based but sparse dok based.
    def __init__(self):
        self.data = {}
        self.block_lookup = {}
        self.blocks = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):

        tup = tuple(val.ravel())

        idx = self.block_lookup.get(tup, len(self.block_lookup))
        self.block_lookup[tup] = idx

        self.data[key] = idx
        self.blocks[idx] = val

    def items(self):
        yield from self.data.items()

class PotentialHamiltonian:
    def __init__(self, lat: CubicLattice):
        self.lattice = lat
        self.potential = BlockReuseMatrix()
        self.matrix = BlockReuseMatrix()
        self.ham = Hamiltonian(lat)


    def __enter__(self):
        self._hopp = {}
        self._pair = {}
        return self._hopp, self._pair

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Store for later
        self.matrix.update(self._hopp)
        self.potential.update(self._pair)

        del self._hopp
        del self._pair

    def __str__(self):
        tot = "----Potential Hamiltonian----\n"
        tot += f" nMatrix: {len(self.matrix)} nPotential: {len(self.potential)}\n"
        tot += "----END----"
        return tot

    def find_blocks(self):
        # Find the permutation
        with self.ham as (H, D):
            for (i, j), val in self.matrix.items():
                H[i, j] = val
            for (i, j), val in self.potential.items():
                D[i, j] = val

        matrix = self.ham.matrix(format='bsr')

        num_blocks, labels = connected_components(csgraph=(matrix != 0).astype(int))
        perm = np.argsort(labels)

        return num_blocks, perm


def main():

    lat = CubicLattice((100, 1, 1))
    ham = PotentialHamiltonian(lat)
    with ham as (H, V):
        for i in lat.sites():
            H[i, i] = -0.5 * sigma0
            x, _, _ = i
            if x < 50:
                V[i, i] = -2.0 * jsigma2

        for i, j in lat.bonds():
            H[i, j] = - 1.0 * sigma0

    ham.find_blocks()

if __name__ == '__main__':
    main()