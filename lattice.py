import numpy as np

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

    def effective_shape(self):
        shape = list(self.shape)
        for dim, period in self.period.items():
            shape[dim] = period
        return tuple(shape)

    def number_of_sites(self) -> int:
        return np.prod(self.effective_shape())


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




def main():

    lat = Lattice((10, 10, 1), period={
        0: 5,
        1: 5,
    })

    # for i, j in lat.bonds():
    #     print(i, j)
    print(lat.number_of_sites())


if __name__ == '__main__':
    main()

