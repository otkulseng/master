import numpy as np
from scipy.sparse import dok_matrix
from tqdm import tqdm
from bdg import *
from chebyshev import cheby_from_dct
from bodge import CubicLattice

from stochastic_trace_estimation import clenshaw_chebyshev_stochastic_estimation


class BlockReuseMatrix:
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
        self.potential = {}
        self.matrix = {}
        self.block_lookup = {}
        self.inverse_block_lookup = {}

    def __enter__(self):
        self._hopp = BlockReuseMatrix()
        self._pair = {}
        return self._hopp, self._pair

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Store for later
        # Vector difference from ri to rj

        for (i, j), val in tqdm(self._hopp.items(), desc="Hopping"):
            # Convert the positions i, j into actual indices
            i = self.lattice.index(i)
            j = self.lattice.index(j)
            self.matrix[(i, j)] = val
        self.block_lookup = self._hopp.block_lookup
        self.inverse_block_lookup = self._hopp.blocks


        for (i, j), val in tqdm(self._pair.items(), desc="Pair"):
            i = self.lattice.index(i)
            j = self.lattice.index(j)

            self.potential[i, j] = val

        del self._hopp
        del self._pair

    def __str__(self):
        tot = "----Potential Hamiltonian----\n"
        tot += f" nEntries: {len(self.matrix)} Blocks: {self.inverse_block_lookup}\n"
        tot += "----END----"
        return tot

    def base_matrix(self):
        indices = np.zeros((len(self.matrix), 3), dtype=np.int32)

        for idx, key in enumerate(self.matrix):
            i, j = key
            k = self.matrix[key]
            indices[idx] = [i, j, k]

        _, val = next(iter(self.inverse_block_lookup.items()))
        shape = tuple([len(self.inverse_block_lookup)] + list(val.shape))
        blocks = np.zeros(shape, dtype=np.complex128)


        for block_index, block in self.inverse_block_lookup.items():
            blocks[block_index] = block

        return bdg_base_matrix(indices, blocks)


    def order_matrix(self):
        indices = torch.zeros((len(self.potential), 2), dtype=torch.long)
        values = torch.zeros(len(self.potential), dtype=torch.float32)

        for idx, key in enumerate(self.potential):
            i, j = key
            indices[idx, 0] = i
            indices[idx, 1] = j
            values[idx] = self.potential[key]

        return bdg_order_matrix_indices(indices), values

    def solve(self, temperature: float = 0):
        # Free energy minimization using chebyshev expansion and stochastic
        # trace estimation.

        # The H_ij and -H_ij^* terms
        base_matrix = self.base_matrix()

        indices, potential = self.order_matrix()


        # Start out with all parameters equal to 1
        order_parameters = torch.ones(potential.size(), requires_grad=True, dtype=torch.float32)
        # Currently, row and col is equal for s-wave



        norm = base_matrix.norm() + 1

        base_matrix = base_matrix / norm


        def free_energy_func(x):
            abs_term = norm * np.abs(x) / 4
            if temperature > 0:
                abs_term += temperature * np.log(1 + np.exp(- norm / temperature * np.abs(x)))
            return -abs_term
            # Above equivalent to T ln(2cosh(norm x / 2T))
            # return - temperature * np.log(2) + norm * np.abs(x)  + temperature * np.log(1 + np.exp(- A * np.abs(x)))


        N: int = 1000
        n_vecs: int = 100
        # TODO: Remove magic number
        coefs = cheby_from_dct(free_energy_func, 2*N)
        print(coefs)
        vecs = torch.randn((n_vecs, base_matrix.size, 2)).to(dtype=torch.complex128)


        optimizer = torch.optim.LBFGS([order_parameters],lr=0.5, history_size=10, max_iter=30)


        def closure():
            optimizer.zero_grad()
            tanh = torch.tanh(order_parameters)
            opm = torch.concat([tanh, -tanh]) / norm

            def inner(v):
                return base_matrix.matvec(v) + apply_order_parameters(
                    indices, opm, v
                )

            trace = clenshaw_chebyshev_stochastic_estimation(
                inner,
                coefs,
                vecs
            )

            # assert(False)
            dot = torch.dot(tanh, tanh / potential)
            total = torch.real(trace + dot)

            print(trace.item(), dot.item(), tanh.mean().item())

            total.backward()

            return total

        for i in range(1):
            optimizer.step(closure)
            current_loss = closure().item()
            print(f"Epoch {i+1:02d}: x = {torch.tanh(order_parameters).mean().item():.6f}, loss = {current_loss:.6f}")

        return torch.tanh(order_parameters)






import matplotlib.pyplot as plt
sigma0 = np.eye(2, dtype=np.float32)
sigma3 = np.array([[1, 0], [0, -1]], dtype=np.float32)
jsigma2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
def main():

    N = 100
    lat = CubicLattice((100, 1, 1))

    np.fft.fftfreq()

    ham = PotentialHamiltonian(lat)

    with ham as (H, V):

        for i, j in tqdm(lat.bonds(), desc="Bond loop"):
            H[i, j] = -1 * sigma0

        for i, j in lat.edges():
            H[i, j] = -1 * sigma0

        for i in tqdm(lat.sites(), desc="Site loop"):

            H[i, i] = - 0.0 * sigma0 #+ (x % 3) * sigma3
            # if x < 50:
            V[i, i] = 2.0

    res = ham.solve(0.1).detach().numpy()
    plt.plot(res)
    plt.savefig("both.pdf")


if __name__ == '__main__':
    main()

