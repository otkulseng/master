from typing import NamedTuple, Any
import jax.numpy as jnp
import jax
from bdg import (
    BDGMatrix,
    self_consistency_equation
)
from typing import Union, Iterable

# from bdg import BDGMatrix, new_bdg_matrix, update_H, update_V, to_dense, StaticBDGMatrix
from util import generate_chebyshev_kmodes, add_diagonal


class CubicLattice:
    def __init__(self, Nx, Ny, Nz):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.index = jax.jit(
            jax.vmap(
                lambda site: site[-1]
                + site[-2] * self.Nz
                + site[-3] * self.Nz * self.Ny
            )
        )

    def size(self):
        return self.Nx * self.Ny * self.Nz

    def sites(self):
        # site (x, y, z) numbered as z + Nz * y + Nz * Ny * x
        return jnp.indices((self.Nx, self.Ny, self.Nz)).reshape((3, -1)).T

    def bonds(self, dim=None):
        sites = self.sites()
        indices = jnp.arange(self.size())
        bonds = []

        # x, y, and z position of the sites
        x = sites[:, 0]
        y = sites[:, 1]
        z = sites[:, 2]

        if self.Nx > 1:
            # Add all bonds in x direction
            mask_x = x < (self.Nx - 1)  # Careful at the end
            src_x = indices[mask_x]
            dst_x = src_x + self.Ny * self.Nz
            bonds += [
                jnp.stack([sites[src_x], sites[dst_x]], axis=1),
                jnp.stack([sites[dst_x], sites[src_x]], axis=1),
            ]

        if self.Ny > 1:
            # Add all bonds in y direction
            mask_y = y < (self.Ny - 1)
            src_y = indices[mask_y]
            dst_y = src_y + self.Nz
            bonds += [
                jnp.stack([sites[src_y], sites[dst_y]], axis=1),
                jnp.stack([sites[dst_y], sites[src_y]], axis=1),
            ]

        if self.Nz > 1:
            # Add all bonds in z direction
            mask_z = z < (self.Nz - 1)
            src_z = indices[mask_z]
            dst_z = src_z + 1
            bonds += [
                jnp.stack([sites[src_z], sites[dst_z]], axis=1),
                jnp.stack([sites[dst_z], sites[src_z]], axis=1),
            ]

        return jnp.concatenate(bonds, axis=0)

    def all(self):
        sites = self.sites()
        stacked_sites = jnp.stack([sites, sites], axis=1)
        return jnp.concatenate([self.bonds(), stacked_sites], axis=0)


class Hamiltonian:
    def __init__(self, lat: CubicLattice, kmodes: Union[int, Iterable[int]]):
        if isinstance(kmodes, int):
            kmodes = [kmodes]

        assert len(kmodes) == 1  # For now, only 1D
        self.num_kmodes = kmodes[0]

        self.kmodes = generate_chebyshev_kmodes(self.num_kmodes)
        self.mask = jnp.tile(jnp.array([1, 1, -1, -1]), lat.size())

        self.lat = lat

        self.H_idx = jnp.empty((0, 2), dtype=jnp.int32)
        self.H_blk = jnp.empty((0, 2, 2), dtype=jnp.complex64)
        self.V_idx = jnp.empty((0, 2), dtype=jnp.int32)
        self.V_blk = jnp.empty(
            (0,), dtype=jnp.complex64
        )  # TODO: Change to (0, 2, 2) s.t. more general

    def get_blocks(self, filter_fn, value_fn):
        batched_filter = jax.jit(
            jax.vmap(lambda i, j: jnp.logical_and.reduce(jnp.array(filter_fn(i, j))))
        )
        batched_value = jax.jit(
            jax.vmap(lambda i, j: jnp.sum(jnp.array(value_fn(i, j)), axis=0))
        )

        bonds = self.lat.all()
        l = bonds[:, 0, :]
        r = bonds[:, 1, :]

        filtered_bonds = bonds[batched_filter(l, r)]
        L = filtered_bonds[:, 0, :]
        R = filtered_bonds[:, 1, :]

        row_idx = self.lat.index(L)
        col_idx = self.lat.index(R)
        blocks = batched_value(L, R)
        return jnp.stack([row_idx, col_idx], axis=1), blocks

    def add_H(self, filter_fn, value_fn):
        idx, blk = self.get_blocks(filter_fn, value_fn)
        self.H_idx = jnp.concatenate([self.H_idx, idx], axis=0)
        self.H_blk = jnp.concatenate([self.H_blk, blk])

    def add_V(self, filter_fn, value_fn):
        idx, blk = self.get_blocks(filter_fn, value_fn)
        self.V_idx = jnp.concatenate([self.V_idx, idx], axis=0)
        self.V_blk = jnp.concatenate([self.V_blk, blk])

    def solve(self, temp: Union[float, Iterable[float]]):
        if isinstance(temp, float):
            temp = [temp]
        temp = jnp.array(temp)

        assert jnp.all(temp >= 0)

        beta = 1 / (1e-15 + temp)

        matr = BDGMatrix(
            self.H_idx, self.H_blk, self.V_idx, self.V_blk, N=self.lat.size()
        )

        results = []

        def forward(x):
            # x of shape (temp, batch, nnz)

            # vmap over batch dimension of x and kmodes
            batch_fn = jax.vmap(self_consistency_equation,in_axes=(None, 0, 0, None))

            # vmap over temp dimension of x and beta
            temp_fn = jax.vmap(batch_fn, in_axes=(None, 0, None, 0))

            return temp_fn(matr, x, self.kmodes, beta)

        # jit function for speed
        func = jax.jit(forward)
        x0 = jnp.ones(
            (beta.size, self.num_kmodes, matr.pot_idx.size), dtype=jnp.complex64
        )  # (Temp, Batch, Pos)
        for it in range(100):
            xn = func(x0)


            diff_delta = jnp.mean(xn - x0, axis=1) # (temp, nnz)
            res = jnp.linalg.norm(diff_delta, axis=-1)
            print(jnp.mean(res))
            if jnp.mean(res) < 1e-3:

                return jnp.mean(xn, axis=1) # (temp, nnz)

            x0 = xn

        return []


# NamedTuple to be a pytree. This is a custom sparse representation of a BDG Matrix

import matplotlib.pyplot as plt


def main():
    # def V(i, j):
    #     return i + j

    sigma0 = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)

    # Create system
    lat = CubicLattice(1, 1, 100)
    ham = Hamiltonian(lat, [101])
    mu = 0.0
    V = 0.8
    ham.add_H(lambda i, j: [jnp.all(i == j)], lambda i, j: [-mu * sigma0])
    ham.add_H(lambda i, j: [jnp.any(i != j)], lambda i, j: [-sigma0])
    ham.add_V(lambda i, j: [jnp.all(i == j), i[2] < 50], lambda i, j: [-V])

    # ham.add_H()

    temps = jnp.linspace(0, 0.03, 3)
    temps = [0.0]
    res = ham.solve(temps)

    print()
    for x, t in zip(res, temps):
        print(x.shape)
        print(x)
        plt.plot(jnp.array(x), label=f"{t}")
    plt.legend()

    plt.savefig("temp.pdf")

    # print("Done")


if __name__ == "__main__":
    main()
