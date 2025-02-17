from bodge import *
from tqdm import tqdm
import torch
from scipy.sparse.csgraph import connected_components
import numpy as np
from scipy.sparse import dok_matrix
from matmul import BlockSparseMatrix
from typing import Union, Iterable
from bdg import DenseBDGSolver
from optimization import broydenB2, broydenB1
from dct import dct, idct

import torch._dynamo
torch._dynamo.config.suppress_errors = True
class BlockReuseMatrix:
    def __init__(self, lat: CubicLattice):
        self.lat = lat
        self.data = dok_matrix((lat.size, lat.size), dtype=np.int32)

        self.block_lookup = {}
        self.blocks = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        i, j = key
        i = self.lat.index(i)
        j = self.lat.index(j)
        tup = tuple(np.ravel(val))

        # 1 indexed to leverage tocoo (which removes block index 0)
        idx = self.block_lookup.get(tup, len(self.block_lookup) + 1)
        self.block_lookup[tup] = idx

        self.data[i, j] = idx
        self.blocks[idx] = val

    def items(self):
        yield from self.data.items()

    def to_tensor(self):
        block_data = np.zeros((len(self.blocks), 2, 2), dtype=np.complex128)

        for idx, blk in self.blocks.items():
            # Remember: 1 indexed
            block_data[idx - 1] = blk

        coo = self.data.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        idx = torch.tensor(coo.data, dtype=torch.long) - 1

        return torch.vstack([row, col, idx]).T, torch.tensor(block_data)


class PotentialMatrix:
    def __init__(self, lat: CubicLattice):
        self.lat = lat
        self.data = dok_matrix((lat.size, lat.size), dtype=np.float32)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        i, j = key
        i = self.lat.index(i)
        j = self.lat.index(j)

        self.data[i, j] = val

    def to_tensor(self):
        coo = self.data.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        data = torch.tensor(coo.data, dtype=torch.float32)
        return torch.vstack([row, col]).T, data


class PotentialHamiltonian:
    def __init__(self, lat: CubicLattice):
        self.lattice = lat
        self._matrix = BlockReuseMatrix(self.lattice)

        self._potential = PotentialMatrix(self.lattice)

    def __enter__(self):
        return self._matrix, self._potential

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def solve(self, kmodes: Union[int, Iterable[int]] = [], cosine_epsilon=1e-3):
        if isinstance(kmodes, int):
            kmodes = [kmodes]

        H_indices, H_blocks = self._matrix.to_tensor()
        V_indices, V_potential = self._potential.to_tensor()

        solver = DenseBDGSolver(
            H_indices,
            H_blocks,
            V_indices,
            V_potential,
            kmodes=torch.tensor(kmodes),
            cosine_threshold=cosine_epsilon
        )
        return solver

    # def matrix(self):
    #     indices, blocks = self._matrix.to_tensor()
    #     indices, blocks = bdg_base_matrix(indices, blocks)

    #     return BlockSparseMatrix(indices, blocks)

    # def eigenvalues(self):
    #     matrix = self.matrix().to_dense()

    #     # eigvalsh stable for gradients
    #     return torch.linalg.eigvalsh(matrix)

import matplotlib.pyplot as plt
import time
def main():
    Nfm = [20]
    Nsc = 20
    results = []

    for n in tqdm(Nfm):
        lat = CubicLattice((int(Nsc + n), 1, 1))
        ham = PotentialHamiltonian(lat)
        with ham as (H, V):
            for i in tqdm(lat.sites()):
                x, _, _ = i
                matr = - 0.01 * sigma0

                if x < Nsc:
                    V[i, i] = -1.0
                else:
                    matr += 0.3 * sigma2
                H[i, i] = matr

            for i, j in tqdm(lat.bonds()):
                H[i, j] = -1.0 * sigma0

        solver = ham.solve(
            kmodes=[250]
        )

        res = solver.solve(0.0)
        plt.plot(res)
        plt.savefig("temp.pdf")

    #     maxval = 1.0
    #     if len(results) > 0:
    #         maxval = results[0]

    #     results.append(solver.critical_temperature(eps=1e-3, maxval=maxval))

    # plt.plot(Nfm, results)
    # plt.savefig("crittemp.pdf")
    # # solver.critical_temperature()


    # plt.plot(x0)
    # plt.show()

if __name__ == "__main__":
    main()
