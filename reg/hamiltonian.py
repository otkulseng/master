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

# from storage import store
import storage
import zarr


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
    def __init__(self, lat: CubicLattice, kmodes: Union[int, Iterable[int]] = []):
        self.lattice = lat
        self._matrix = BlockReuseMatrix(self.lattice)
        self._potential = PotentialMatrix(self.lattice)


        if isinstance(kmodes, int):
            kmodes = [kmodes]
        self.kmodes = kmodes


        # This Hamiltonian may have order parameters.
        # Little bit of everything?
        # All corr stored at the same place
        # All order parameters stored at the same place
        # data/finish/neel
        # data/finish/SF

        # Critical Temperature
        #   Metadata
        # Lattice
        # Name (as a substitute for the entire matrix)
        #

        # data/finish/corr
        # data/finish/
        # data/finish/
        #   corr
        #   order
        #


    def __enter__(self):
        return self._matrix, self._potential

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def solver(self):
        H_indices, H_blocks = self._matrix.to_tensor()
        V_indices, V_potential = self._potential.to_tensor()
        return DenseBDGSolver(
            H_indices,
            H_blocks,
            V_indices,
            V_potential,
            kmodes=torch.tensor(self.kmodes),
        )


    def critical_temperature(self):
        return self.solver().critical_temperature()

    def condensation_energy(self, x: torch.Tensor, ):
        pass

    def solve(self, temperature=0.0):
        return self.solver().solve(temperature)
        # return solver.solve(temperature)

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
    Nfm = [50]
    Nsc = 50
    kmodes = [200]
    mu = -0.1
    pot = 1.0
    m = 0.25

    storage.new("FM")
    for n in tqdm(Nfm):
        shape = (int(Nsc + n), 1, 1)

        storage.save_kwargs(
            name='config',
            Nsc=Nsc,
            Nfm = n,
            shape=shape,
            kmodes=kmodes,
            mu=mu,
            V = pot,
            m=m
        )

        lat = CubicLattice(shape)
        ham = PotentialHamiltonian(lat, kmodes=kmodes)
        with ham as (H, V):
            for i in tqdm(lat.sites()):
                x, _, _ = i
                matr = - mu * sigma0

                if x < Nsc:
                    V[i, i] = - pot
                else:
                    matr += m * sigma1
                H[i, i] = matr

            for i, j in tqdm(lat.bonds()):
                H[i, j] = -1.0 * sigma0

        # T = ham.critical_temperature()
        # x = ham.solve(0.0)
        solver = ham.solver()

        T = solver.critical_temperature()


        # ham.crit*


    # storage.close()


if __name__ == "__main__":
    main()
