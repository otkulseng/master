import jax
import jax.numpy as jnp
from util import insert_blocks
from typing import NamedTuple
from bdg import make_bdg_H_term, make_bdg_D_term
from optimization import broydenb2, stable_newton
# from mpi4py import MPI


import storage
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class CubicLattice:
    def __init__(self, shape: tuple[int], periodic: tuple[bool]):
        self.shape = shape
        self.periodic = periodic

        Nx, Ny, Nz = shape
        self.size = Nx * Ny * Nz

    def tree_flatten(self):
        children = ()
        aux_data = (self.shape, self.periodic)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the NamedTuple from the flattened children and the static data
        # (sites, bonds) = children
        (shape, periodic) = aux_data
        return cls(shape, periodic)


@jax.jit
def lattice_sites(sys: CubicLattice):
    Nx, Ny, Nz = sys.shape
    return jnp.indices((Nx, Ny, Nz)).reshape((3, -1)).T


@jax.jit
def lattice_size(sys: CubicLattice):
    Nx, Ny, Nz = sys.shape
    return Nx * Ny * Nz


@jax.jit
def lattice_bonds(sys: CubicLattice):
    Nx, Ny, Nz = sys.shape
    px, py, pz = sys.periodic
    bonds = []

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                l = [i, j, k]
                r = l

                if i > 0 or px:
                    r = [(Nx + i - 1) % Nx, j, k]
                if j > 0 or py:
                    r = [i, (Ny + j - 1) % Ny, k]
                if k > 0 or pz:
                    r = [i, j, (Nz + k - 1) % Nz]

                bonds.append([l, r])
                bonds.append([r, l])

    return jnp.array(bonds)


def vmap_decorator(in_axes=0, out_axes=0, axis_name=None):
    def actual_decorator(fn):
        return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name)

    return actual_decorator


@jax.jit
@vmap_decorator(in_axes=(None, 0))
def lattice_index(sys: CubicLattice, point):
    Nx, Ny, Nz = sys.shape
    return point[-1] + point[-2] * Nz + point[-3] * Nz * Ny


# @jax.jit
def empty_matrix(sys: CubicLattice):
    N = sys.size
    return jnp.zeros((4 * N, 4 * N), dtype=jnp.complex128)


def bdg_add_H(matr: jax.Array, l: jax.Array, r: jax.Array, val: jax.Array):
    """_summary_

    Args:
        matr (jax.Array): 4N x 4N
        l (jax.Array): numel,
        r (jax.Array): numel,
        val (jax.Array): numel(1), 2, 2
    """

    numel = l.shape[0]
    val = jnp.broadcast_to(val, (numel, 2, 2))
    idx = jnp.stack([l, r], axis=0).transpose((-1, -2))

    new_idx, new_val = make_bdg_H_term(idx, val)
    return insert_blocks(matr, new_idx, new_val)


def bdg_add_D(matr: jax.Array, l: jax.Array, r: jax.Array, val: jax.Array):
    numel = l.shape[0]

    jsigma2 = jnp.array([[0, 1], [-1, 0]], dtype=jnp.complex128)

    val = jnp.broadcast_to(val[:, None, None] * jsigma2[None, :, :], (numel, 2, 2))
    idx = jnp.stack([l, r], axis=0).transpose((-1, -2))

    new_idx, new_val = make_bdg_D_term(idx, val)
    return insert_blocks(matr, new_idx, new_val)


@jax.jit
def temperature_independent_correlations(Q: jax.Array):
    """
    Args:
        Q (jax.Array): (4N, 4N)
    """

    size = Q.shape[0]
    N = size // 4

    # Now, Q[n, :] is eigenvector corresponding to eigenvalue E_n
    Q = jnp.swapaxes(Q, -1, -2)

    # (Eigenvalue, Position, Nambu)
    Q = jnp.reshape(Q, shape=(Q.shape[-2], N, 4))

    uup = Q[..., 0]
    udo = Q[..., 1]
    vup = Q[..., 2]
    vdo = Q[..., 3]

    return (udo.conj() * vup - uup.conj() * vdo) / 2  # (Eigenvalue, Position)


@jax.jit
def tanhify_eigenvalues(L: jax.Array, beta):
    # L (N), beta (float)
    return jnp.tanh(jnp.real(beta * L / 2))


def consistency(L, Q, idx, V, t):
    corr = temperature_independent_correlations(Q)  # (Eigenvalue, Position) : (4N, N)
    tanh = tanhify_eigenvalues(L, 1 / (1e-10 + t))  # (Eigenvalue, ) : (4N,)

    res: jax.Array = tanh[:, None] * corr[:, idx] * V[None, :]  # (Eigenvalues, nnz)

    # Keep only positive eigenvalues
    size = res.shape[0]
    res = res[size // 2 :, :]

    # Sum over eigenvalues
    return jnp.sum(res, axis=0)  # (nnz)


def jacobian(L: jax.Array, Q: jax.Array, idx: jax.Array, V, t):
    K = jnp.array(
        [
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=Q.dtype,
    )
    size = L.shape[0]
    N = size // 4
    nnz = idx.shape[0]

    out = jnp.zeros((nnz, nnz), dtype=jnp.complex128)

    # p0 = consistency(L, Q, idx, V, t)

    Q = Q.transpose((-1, -2))
    Q_b = Q.reshape((size, N, 4))[:, idx, :]  # (4N, nnz, 4)

    denom = jnp.expand_dims(L, -1) - jnp.expand_dims(L, -2)
    denom = jnp.where(jnp.abs(denom) < 1e-10, jnp.inf, denom)

    tanh = tanhify_eigenvalues(L, 1 / (1e-10 + t))  # (Eigenvalue, ) : (4N,)

    Qouter = jnp.reshape(Q, shape=(Q.shape[-2], N, 4))

    uup0 = Qouter[..., 0]
    udo0 = Qouter[..., 1]
    vup0 = Qouter[..., 2]
    vdo0 = Qouter[..., 3]
    for j in range(nnz):
        # Step 1, calculate change in L and Q using perturbation theory
        cur = Q_b[:, j, :]
        numerator = cur.conj() @ K @ cur.T

        # Ldiff = jnp.diagonal(numerator)

        factor = numerator / denom

        Qdiff = jnp.einsum("ij, ik->jk", factor, Q)

        # (Eigenvalue, Position, Nambu)
        Qinner = jnp.reshape(Qdiff, shape=(Q.shape[-2], N, 4))

        uup = Qinner[..., 0]
        udo = Qinner[..., 1]
        vup = Qinner[..., 2]
        vdo = Qinner[..., 3]

        # corr = ((udo0 + udo).conj() * (vup0 + vup) - (uup0 + uup).conj() * (vdo0 + vdo)) / 2

        A = udo0.conj() * vup + udo.conj() * vup0
        B = uup0.conj() * vdo + uup.conj() * vdo0
        corr = (A - B) / 2

        res: jax.Array = tanh[:, None] * corr[:, idx] * V[None, :]  # (Eigenvalues, nnz)

        # Keep only positive eigenvalues
        size = res.shape[0]
        res = res[size // 2 :, :]

        # Sum over eigenvalues
        p = jnp.sum(res, axis=0)  # (nnz)

        # p = consistency(L, (Q + Qdiff).transpose((-1, -2)), idx, V, t)

        out = out.at[j].set(p)
    return out


def cartesian_product(*arrays):
    grids = jnp.meshgrid(*arrays, indexing="ij")
    return jnp.stack(grids, axis=-1).reshape(-1, len(arrays))


# @jax.jit
def order_parameters(lat: CubicLattice, r, k, t, mu, V, m):
    """_summary_

    Args:
        lat (CubicLattice): The lattice on which
        mu (_type_): _description_
        k (_type_): _description_
        V (_type_): _description_

    Returns:
        _type_: _description_
    """
    Nx, Ny, Nz = lat.shape
    sites = lattice_sites(lat)  # (N, 3)
    bonds = lattice_bonds(lat)  # (N, 2, 3)

    delta_sites = sites[sites[:, 1] == 1]

    sigma0 = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128)
    sigmax = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)


    site_indices = lattice_index(lat, sites)
    delta_indices = lattice_index(lat, delta_sites)
    bonds_l_indices = lattice_index(lat, bonds[:, 0, :])
    bonds_r_indices = lattice_index(lat, bonds[:, 1, :])

    top_layer_sites = lattice_index(lat, sites[sites[:, 1] == 2])
    bottom_layer_sites = lattice_index(lat, sites[sites[:, 1] == 0])


    r_sites = lattice_index(lat, sites[jnp.logical_and(sites[:, 1] == 0, sites[:, 0] < r)])
    # @jax.jit
    def matrix(x, k0, mu0, m0):
        # Create empty matrix (zeros) of the correct size
        matr = empty_matrix(lat)

        # Add hopping
        matr = bdg_add_H(
            matr, bonds_l_indices, bonds_r_indices, -1.0 * sigma0[None, ...]
        )  # Broadcast

        # Add top layer magnetic field
        matr = bdg_add_H(
            matr, top_layer_sites, top_layer_sites , -m0 * sigmax[None, ...]
        )

        # Add bottom layer magnetic field
        matr = bdg_add_H(
            matr, bottom_layer_sites, bottom_layer_sites , +m0 * sigmax[None, ...]
        )

        # Add r-dependence
        matr = bdg_add_H(
            matr, r_sites, r_sites , -2*m0 * sigmax[None, ...]
        )

        # Add k-val and mu
        matr = bdg_add_H(
            matr,
            site_indices,
            site_indices,
            -(mu0 + 2 * jnp.cos(k0)) * sigma0[None, ...],
        )

        # Insert deltas
        matr = bdg_add_D(matr, delta_indices, delta_indices, -x)

        # Diagonalize
        return matr

    def free_energy(x, k0, t0, mu0, V0, m0):
        L = jnp.linalg.eigvalsh(matrix(x,  k0, mu0, m0))

        # Keep only positive
        L = L[L.shape[0] // 2 :]

        # Superconducting contribution
        E0 = jnp.sum(x.conj() * x / V0)

        # Non-entropic contribution
        H0 = -1 / 2 * jnp.sum(L)

        # Entropy
        S = jnp.sum(jnp.log(1 + jnp.exp(-(1 / (1e-10 + t0)) * L)))

        return jnp.real(E0 + H0 - t0 * S)

    def condensation_energy(x, k0, t0, mu0, V0, m0):
        return free_energy(x, k0, t0, mu0, V0, m0) - free_energy(
            jnp.zeros_like(x), k0, t0, mu0, V0, m0
        )

    def iteration_step(x, k0, t0, mu0, V0, m0):
        L, Q = jnp.linalg.eigh(matrix(x, k0, mu0, m0))

        V0 = jnp.array([V0])
        xnext = consistency(L, Q, delta_indices, V0, t0)
        jac = jacobian(L, Q, delta_indices, V0, t0)

        return xnext - x, jac - jnp.eye(delta_indices.shape[0], dtype=x.dtype)

    vmap_iteration_step = jax.jit(jax.vmap(iteration_step))

    vmap_condensation_energy = jax.jit(jax.vmap(condensation_energy))


    def solve(tuples: jax.Array):
        x0 = jnp.ones((tuples.shape[0], delta_sites.shape[0]), dtype=jnp.complex128)

        def forward(x, mask):
            tup = tuples[mask]
            return vmap_iteration_step(x, *tup.T)

            # return jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)

        # forw = jax.jit(forward)

        res = stable_newton(forward, x0, max_iter=100)
        storage.store(["order_params", "tol", "points"], [res.x, res.fx, tuples])
        B, _ = tuples.shape

        print("Calculating condensation energy")
        cond_energy = vmap_condensation_energy(res.x, *tuples.T).reshape((B, 1))
        combined = jnp.concatenate([tuples, jnp.ones_like(cond_energy)*r, cond_energy], axis=-1)
        storage.store(['condensation_energy'], [combined])

    tuples = cartesian_product(k, t, mu, V, m)

    # Only keep the tuples corresponding to this rank
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    # tuples = tuples[rank::size]

    # asd = [i for i in range(100)]

    # n_tasks = 10
    # n_cpu_per_task = 10

    batch_size = 100

    for i in tqdm(range(0, len(tuples), batch_size)):
        min_idx = i
        max_idx = min(i + batch_size, len(tuples))
        try:
            solve(tuples[min_idx:max_idx])
            jax.clear_caches()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            continue


def main():

    sys = CubicLattice((30, 3, 1), (True, False, False))
    storage.init("other30")

    # 200 per cpu time, 2000 per time x 60 rekker ca 100000

    # 100 temps
    # 100 diag_vals
    # 10 000 values that I must have
    # In 10000 hours, I can solve at least 1 000 000, if not a lot more (?)
    # which means 10 000 divided through
    # V, m and r0
    # r0 = 30 vals
    # 
    N = 30
    for r0 in [0, 1]:
        x = order_parameters(
            sys,
            r = r0,
            k=jnp.pi * (2 * jnp.arange(N) + 1) / (2 * N),
            # t=jnp.linspace(0.0, 0.05, 50),
            t = jnp.linspace(0, 0.05, N),
            # mu = jnp.linspace(0.0, 0.2, 4),
            mu = jnp.array([0.1]),
            V = jnp.array([0.8]),
            # V=jnp.linspace(0.4, 0.8, num_temp),
            # V = jnp.linspace(0.4, 0.8, 4),
            m=jnp.array([0.1]),
        )


    # print(f"Number of iterations: {aux.iterations}")
    # matr = x.reshape((N, num_temp, -1)).mean(0)

    # import matplotlib.pyplot as plt

    # for i in range(num_temp):
    #     plt.plot(matr[i, :], label=f"{i}")
    # plt.legend()
    # plt.savefig("temp.pdf")

    # print(matr)
    # print(matr.shape)


if __name__ == "__main__":
    main()
