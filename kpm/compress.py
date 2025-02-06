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
        blocks = np.zeros(shape, dtype=np.float32)


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

        return indices, values

    def solve(self, temperature: float = 0):
        # Free energy minimization using chebyshev expansion and stochastic
        # trace estimation.

        # The H_ij and -H_ij^* terms
        base_matrix = self.base_matrix()

        indices, potential = self.order_matrix()


        # Start out with all parameters equal to 1
        order_parameters = torch.zeros(indices[:, 0].size(), requires_grad=True, dtype=torch.float32)
        # Currently, row and col is equal for s-wave
        indices = bdg_order_matrix_indices(indices)


        norm = base_matrix.norm()



        base_matrix = base_matrix / norm


        def free_energy_func(x):
            abs_term = norm * np.abs(x) / 2
            if temperature > 0:
                abs_term += temperature * np.log(1 + np.exp(- norm / temperature * np.abs(x)))
            return -abs_term / 2
            # Above equivalent to T ln(2cosh(norm x / 2T))
            # return - temperature * np.log(2) + norm * np.abs(x)  + temperature * np.log(1 + np.exp(- A * np.abs(x)))


        N: int = 100
        n_vecs: int = 500
        # TODO: Remove magic number
        coefs = cheby_from_dct(free_energy_func, 2*N)
        print(coefs)

        vecs = torch.randn((n_vecs, base_matrix.size, 2), dtype=torch.float32)



        optimizer = torch.optim.LBFGS([order_parameters], history_size=10, max_iter=50)


        def closure():
            optimizer.zero_grad()
            tanh = torch.tanh(order_parameters)
            opm = torch.concat([tanh, -tanh])



            def inner(v):
                return base_matrix.matvec(v) + apply_order_parameters(
                    indices, opm, v
                )

            def alpha(v):
                return  (2 * inner(inner(v)) - v)
            def beta(v):
                return -v

            b_prev_prev = torch.clone(vecs)
            b_prev = alpha(b_prev_prev)

            total = torch.einsum(
                "...ij, ...ij->...",
                vecs, b_prev_prev
            )* coefs[0]

            total += torch.einsum(
                "...ij, ...ij->...",
                vecs, b_prev
            ) * coefs[1]

            for k in range(2, N):
                temp =2 * alpha(b_prev) - b_prev_prev
                b_prev_prev = b_prev
                b_prev = temp

                prod = torch.einsum(
                    "...ij, ...ij->...",
                    vecs, b_prev
                )
                print(f"K= {k}: prod: {prod.mean().item()}, coef: {coefs[k]}, vecs: {vecs.abs().mean().item()}")

                total += prod * coefs[k]

            # assert(False)
            dot = torch.dot(tanh, tanh / potential)
            total = total.mean()

            print(total.item(), dot.item())

            total = torch.real(total + dot)
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
    lat = Lattice((100, 1, 1))


    ham = PotentialHamiltonian(lat)

    with ham as (H, V):
        for i in tqdm(lat.sites(), desc="Site loop"):
            x, y, z = i

            H[i, i] = - 1.0 * sigma0 #+ (x % 3) * sigma3
            if x < 50:
                V[i, i] = -2.0

        for i, j in tqdm(lat.bonds(), desc="Bond loop"):
            H[i, j] = -1 * sigma0



    res = ham.solve(0).detach().numpy()
    plt.plot(res)
    plt.savefig("both.pdf")


if __name__ == '__main__':
    main()

