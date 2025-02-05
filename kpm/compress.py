import numpy as np
from scipy.sparse import dok_matrix
from tqdm import tqdm
from bdg import *
from chebyshev import cheby_from_dct

class Lattice:
    def __init__(self, shape: tuple, period: dict[int, int] = {}):
        self.shape = shape
        assert(len(self.shape) == 3)
        assert(len(self.shape) <= 3)
        assert(type(shape) is tuple)

        self.period = period
        for dim, p in period.items():
            if dim < 0 or dim >= 3:
                raise RuntimeError("Dimension {dim} of period not in interval [0, 3)")

            if shape[dim] < p:
                raise RuntimeError("Period cannot be smaller than shape")

    def kmodes(self):
        kmodes =  [
            np.array([0]), np.array([0]), np.array([0])
        ]

        for dim, period in self.period.items():
            kmodes[dim] = period * 2 * np.pi * np.fft.fftfreq(self.shape[dim])
        return kmodes

    def effective_shape(self):
        shape = list(self.shape)
        for dim, period in self.period.items():
            shape[dim] = period
        return tuple(shape)

    def number_of_sites(self) -> int:
        return np.prod(self.effective_shape())

    def index(self, idx) -> int:
        Lx, Ly, _ = self.shape
        i, j, k = idx

        return i + j * Lx + k * Lx * Ly

    def sites(self):
        Lx, Ly, Lz = self.effective_shape()
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    yield (x, y, z)

    def bonds(self):
        # First iterate over all conventional bonds
        Lx, Ly, Lz = self.effective_shape()
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    if x > 0:
                        yield (x-1, y, z), (x, y, z)
                        yield (x, y, z), (x-1, y, z)

                    if y > 0:
                        yield (x, y-1, z), (x, y, z)
                        yield (x, y, z), (x, y-1, z)

                    if z > 0:
                        yield (x, y, z-1), (x, y, z)
                        yield (x, y, z), (x, y, z-1)



    def edges(self, axes: list[int] = []):
        # All conventional edges
        Lx, Ly, Lz = self.effective_shape()
        for ax in axes:
            if ax in self.period:
                continue

            match ax:
                case 0:
                    for y in range(Ly):
                        for z in range(Lz):
                            yield (0, y, z), (Lx-1, y, z)
                            yield (Lx-1, y, z), (0, y, z)
                case 1:
                    for x in range(Lx):
                        for z in range(Lz):
                            yield (x, 0, z), (x, Ly-1, z)
                            yield (x, Ly-1, z), (x, 0, z)
                case 2:
                    for x in range(Lx):
                        for y in range(Ly):
                            yield (x, y, 0), (x, y, Lz-1)
                            yield (x, y, Lz-1), (x, y, 0)


        for dim in self.period:
            match dim:
                case 0:
                    for y in range(Ly):
                        for z in range(Lz):
                            yield (Lx, y, z), (Lx-1, y, z)
                            yield (Lx-1, y, z), (Lx, y, z)
                case 1:
                    for x in range(Lx):
                        for z in range(Lz):
                            yield (x, Ly, z), (x, Ly-1, z)
                            yield (x, Ly-1, z), (x, Ly, z)
                case 2:
                    for x in range(Lx):
                        for y in range(Ly):
                            yield (x, y, Lz), (x, y, Lz-1)
                            yield (x, y, Lz-1), (x, y, Lz)

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
    def __init__(self, lat: Lattice):

        self.lattice = lat
        N = self.lattice.number_of_sites()
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

        return BlockSparseMatrix(indices, blocks)


    def order_matrix(self):
        indices = torch.zeros((len(self.potential), 2), dtype=torch.long)
        values = torch.zeros(len(self.potential), dtype=torch.float64)

        for idx, key in enumerate(self.potential):
            i, j = key
            indices[idx, 0] = i
            indices[idx, 1] = j
            values[idx] = self.potential[key]

        return indices, values

    def solve(self, temperature: float = 0):
        # Free energy minimization using chebyshev expansion and stochastic
        # trace estimation.

        # The H_ij and -H_ij^* terms
        base_matrix = self.base_matrix()

        indices, potential = self.order_matrix()


        # Start out with all parameters equal to 1
        order_parameters = torch.ones(indices[:, 0].size(), requires_grad=True, dtype=torch.complex128)

        # Currently, row and col is equal for s-wave
        indices = bdg_order_matrix_indices(indices)


        norm = base_matrix.norm()
        print(norm)

        base_matrix = base_matrix / norm


        def free_energy_func(x):
            abs_term = norm * np.abs(x) / 2
            if temperature > 0:
                abs_term += temperature * np.log(1 + np.exp(- norm / temperature * np.abs(x)))
            return -abs_term
            # Above equivalent to T ln(2cosh(norm x / 2T))
            # return - temperature * np.log(2) + norm * np.abs(x)  + temperature * np.log(1 + np.exp(- A * np.abs(x)))


        N: int = 101
        n_vecs: int = 1000
        # TODO: Remove magic number
        coefs = cheby_from_dct(free_energy_func, 2*N)

        vecs = torch.randn((n_vecs, base_matrix.size, 2), dtype=torch.complex128)



        optimizer = torch.optim.LBFGS([order_parameters], history_size=10, max_iter=5)


        def closure():
            optimizer.zero_grad()
            opm = torch.concat([order_parameters, -order_parameters.conj()])
            def inner(v):
                return base_matrix.matvec(v) + apply_order_parameters(
                    indices, opm, v
                )

            def alpha(v):
                return 2 * (2 * inner(inner(v)) - v)
            def beta(v):
                return -v

            b_next = torch.zeros_like(vecs)
            b_next_next = torch.zeros_like(vecs)
            b_curr = torch.zeros_like(vecs)

            for k in range(N-1, -1, -1):
                b_curr = coefs[k] * vecs +  alpha(b_next) + beta(b_next_next)

                if k > 0:
                    b_next_next = b_next
                    b_next = b_curr

            b0 = b_curr
            b1 = b_next
            b2 = b_next_next
            res = coefs[0]*vecs + alpha(b1) / 2 + beta(b2)
            res = torch.einsum(
                "...ij, ...ij->...",
                vecs.conj(), res
            ).mean() + torch.real(torch.dot(order_parameters / potential, order_parameters))
            torch.real(res).backward()


            return res

        for i in range(50):
            loss = optimizer.step(closure)
            current_loss = closure().item()
            print(f"Epoch {i+1:02d}: x = {order_parameters.mean().item():.6f}, loss = {current_loss:.6f}")








import matplotlib.pyplot as plt
sigma0 = np.eye(2, dtype=np.complex128)
sigma3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
jsigma2 = np.array([[0, 1], [-1, 0]], dtype=np.complex128)
def main():
    lat = Lattice((10, 1, 1))


    ham = PotentialHamiltonian(lat)

    with ham as (H, V):
        for i in tqdm(lat.sites(), desc="Site loop"):
            x, y, z = i

            H[i, i] = - 2.0 * sigma0 #+ (x % 3) * sigma3
            if x < 50:
                V[i, i] = -0.9

        for i, j in tqdm(lat.bonds(), desc="Bond loop"):
            H[i, j] = -1.0 * sigma0


    # print(indices)
    # print(blocks)


    ham.solve()



if __name__ == '__main__':
    main()

