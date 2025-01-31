import numpy as np
from scipy.sparse import coo_matrix
from itertools import product


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
        Lx, Ly, Lz = self.effective_shape()
        i, j, k = idx

        i %= Lx
        j %= Ly
        k %= Lz

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

class PositionBlockCoo:
    # TODO: Make more efficient than this mess
    def __init__(self):
        self.row_indices = []
        self.col_indices = []
        self.data = []
        self.vectors = []

    def add(self, row, col, vector, data):
        self.row_indices.append(row)
        self.col_indices.append(col)
        self.vectors.append(vector)
        self.data.append(data)

    def tocoo(self):
        blocksize = np.array(self.data[0]).size
        data = np.ravel(self.data)
        row = np.repeat(self.row_indices, blocksize)
        col = np.repeat(self.row_indices, blocksize)

        # print(row.dtype, row.shape)
        # print(col.dtype, row.shape)
        return coo_matrix((data, (row, col)))


def block_coo_transform(data, row, col) -> coo_matrix:
    # Would like to coo_matrix((data, (row, col))),
    # but cannot as data is multidimensional. This function fixes that

    yblock_shape = data.shape[2]
    xblock_shape = data.shape[1]

    prod = np.array(list(product(np.arange(xblock_shape), np.arange(yblock_shape))))
    xoffsets = prod[:, 0]
    yoffsets = prod[:, 1]

    row = row * xblock_shape
    col = col * yblock_shape

    row = row[:, np.newaxis] + xoffsets[np.newaxis, :]
    col = col[:, np.newaxis] + yoffsets[np.newaxis, :]

    matr =  coo_matrix(
        (np.ravel(data), (np.ravel(row), np.ravel(col)))
    )

    matr.eliminate_zeros()
    return matr


class PotentialHamiltonian:
    def __init__(self, lat: Lattice):

        self.lattice = lat
        self.matrix = PositionBlockCoo()
        self.potential = PositionBlockCoo()

    def __enter__(self):
        self._hopp = {}
        self._pair = {}
        return self._hopp, self._pair

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Store for later
        # Vector difference from ri to rj
        for (i, j), val in self._hopp.items():
            vec = np.array(j) - np.array(i)
            i = self.lattice.index(i)
            j = self.lattice.index(j)

            self.matrix.add(i, j, vec, val)


        for (i, j), val in self._pair.items():
            vec = np.array(j) - np.array(i)
            i = self.lattice.index(i)
            j = self.lattice.index(j)
            self.potential.add(i, j, vec, val)

        del self._hopp
        del self._pair

    def free_energy(self, temperature: float = 0):

        # Before k-modes, find all parameters needed to specify.
        # They are self.matrix: (i, j, )


        kx, ky, kz = self.lattice.kmodes()
        momentum_states = np.array(list(product(kx, ky, kz)))
        for kvec in momentum_states:
            # val is changed by the cosine
            cosk = np.cos(
                np.dot(self.matrix.vectors,  kvec)
            )
            H = cosk[:, np.newaxis, np.newaxis] * self.matrix.data

            row, col = np.array(self.matrix.row_indices), np.array(self.matrix.col_indices)
            row = row * 2
            col = col * 2
            row = np.concatenate([row, row+1])
            col = np.concatenate([col, col+1])
            data = np.concatenate([H, -np.conj(H)])
            base_matrix = block_coo_transform(data, row, col).tocsr()







sigma0 = np.eye(2)
jsigma2 = np.array([[0, 1], [-1, 0]])
def main():
    lat = Lattice((10, 1, 100), period={
        0: 1,
        # 1: 1,
    })

    ham = PotentialHamiltonian(lat)
    with ham as (H, V):
        for i in lat.sites():
            H[i, i] = - 0.03 * sigma0
            V[i, i] = -0.9 * jsigma2

        for i, j in lat.bonds():
            H[i, j] = -sigma0

        for i, j in lat.edges(axes=[0, 1]):
            H[i, j] = -sigma0

    ham.free_energy(0.4)



if __name__ == '__main__':
    main()

