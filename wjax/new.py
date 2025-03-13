import jax
import jax.numpy as jnp
from util import insert_blocks
from typing import NamedTuple
from bdg import make_bdg_H_term, make_bdg_D_term
from optimization import broydenb2

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


def jacobianv1(L: jax.Array, Q: jax.Array, idx: jax.Array, V, t):
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

    Q = Q.transpose((-1, -2))  # Now, Q[Eigenvalue, Vector] : (4N, 4N)
    Q_smaller = Q.reshape((size, N, 4))[:, idx, :]

    uup_0 = Q_smaller[-2 * N, :, 0]  # (2N, nnz)
    udo_0 = Q_smaller[-2 * N, :, 1]
    vup_0 = Q_smaller[-2 * N, :, 2]
    vdo_0 = Q_smaller[-2 * N, :, 3]

    denom = jnp.expand_dims(L, -2) - jnp.expand_dims(L, -1)
    denom = jnp.where(jnp.abs(denom) < 1e-10, jnp.inf, denom)

    tanhe = tanhify_eigenvalues(L[-2 * N :], 1 / (1e-10 + t))

    for i in range(nnz):
        # calculate this row-wise
        # Step 1, how does the eigenvectors change
        # Q of shape (4N, N, 4). Last N4 elements are the actual vectors
        Q_cur = Q_smaller[:, i, :]
        factor = Q_cur.conj() @ K @ Q_cur.T / denom  # (4N, 4N)

        Q_diff = factor.T @ Q  # (4N, 4N)
        # print(Q_diff)

        Q_diff = Q_diff.reshape(size, N, 4)[-2 * N :, idx, :]
        uup_diff = Q_diff[..., 0]
        udo_diff = Q_diff[..., 1]
        vup_diff = Q_diff[..., 2]
        vdo_diff = Q_diff[..., 3]

        res = jnp.sum(  # First has shape (2N, nnz) * (2N, None) * (None, nnz)
            (
                (udo_0.conj() * vup_diff + udo_diff.conj() * vup_0)
                - (uup_0.conj() * vdo_diff + uup_diff.conj() * vdo_0)
            )
            / 2
            * tanhe[:, None]
            * V[None, :],
            axis=0,
        )

        out = out.at[i].set(res)
    return -out


def jacobianv0(L: jax.Array, Q: jax.Array, idx: jax.Array, V, t):
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

    Q = Q.transpose((-1, -2)).reshape(
        size, N, 4
    )  # Now, Q[n, :] is eigenvector corresponding to L[n]

    # Need to calculate each perturbation multiplication
    Q_nnz = Q[:, idx, :]  # (4N, nnz, 4)

    K_Q_nnz = jnp.einsum("ijk, kl->lji", Q_nnz, K)  # (4, nnz, 4N)

    denom = jnp.expand_dims(L, -1) - jnp.expand_dims(L, -2)
    denom = jnp.where(jnp.abs(denom) < 1e-10, jnp.inf, denom)

    out = jnp.zeros((nnz, nnz), dtype=Q.dtype)

    uup_0 = Q_nnz[-2 * N :, :, 0]
    udo_0 = Q_nnz[-2 * N :, :, 1]
    vup_0 = Q_nnz[-2 * N :, :, 2]
    vdo_0 = Q_nnz[-2 * N :, :, 3]

    tanhe = tanhify_eigenvalues(L[-2 * N :], 1 / (1e-10 + t))

    for i in range(nnz):
        cur_Q = Q_nnz[:, i, :]  # (4N, 4)
        cur_K_Q_nnz = K_Q_nnz[:, i, :]  # (4, 4N)
        Q_K_Q = jnp.matmul(cur_Q.conj(), cur_K_Q_nnz)  # (4N, 4N)
        Q_K_Q = Q_K_Q / denom  # (4N, 4N)

        Q_diff = jnp.einsum("ij, jkl -> ikl", Q_K_Q, Q_nnz)[
            -2 * N :, :, :
        ]  # (2N, nnz, 4)
        uup_diff = Q_diff[..., 0]
        udo_diff = Q_diff[..., 1]
        vup_diff = Q_diff[..., 2]
        vdo_diff = Q_diff[..., 3]

        res = jnp.sum(  # First has shape (2N, nnz) * (2N, None) * (None, nnz)
            (
                (udo_0.conj() * vup_diff + udo_diff.conj() * vup_0)
                - (uup_0.conj() * vdo_diff + uup_diff.conj() * vdo_0)
            )
            / 2
            * tanhe[:, None]
            * V[None, :],
            axis=0,
        )

        out = out.at[i].set(res)

    return out


def cartesian_product(*arrays):
    grids = jnp.meshgrid(*arrays, indexing="ij")
    return jnp.stack(grids, axis=-1).reshape(-1, len(arrays))


# @jax.jit
def order_parameters(lat: CubicLattice, mu, k, V, t, max_iter=50, eps=1e-7):
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

    delta_sites = sites[sites[:, 0] < Nx // 2]

    sigma0 = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128)

    site_indices = lattice_index(lat, sites)
    delta_indices = lattice_index(lat, delta_sites)
    bonds_l_indices = lattice_index(lat, bonds[:, 0, :])
    bonds_r_indices = lattice_index(lat, bonds[:, 1, :])

    # @jax.jit
    def iteration_step(x, mu0, k0, V0, t0):
        # Create empty matrix (zeros) of the correct size
        matr = empty_matrix(lat)

        # Add hopping
        matr = bdg_add_H(
            matr, bonds_l_indices, bonds_r_indices, -1.0 * sigma0[None, ...]
        )  # Broadcast

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
        L, Q = jnp.linalg.eigh(matr)

        V0 = jnp.array([V0])
        xnext = consistency(L, Q, delta_indices, V0, t0)
        jac = jacobian(L, Q, delta_indices, V0, t0)

        return xnext - x, jac - jnp.eye(delta_indices.shape[0], dtype=x.dtype)

        # return jacobian_both(L, Q, delta_indices, V0, t0)

        # return consistency(L, Q, delta_indices, V0, t0) - x, jacobian(
        #     L, Q, delta_indices, V0, t0
        # )

    vmap_iteration_step = jax.jit(jax.vmap(iteration_step))

    def forward_mask(x, mask=None):
        if mask is None:
            mask = jnp.ones(x.shape[0], dtype=bool)
        return vmap_iteration_step(x[mask, :], *tuples[mask, :].T)

    def solve(tuples: jax.Array, x0: jax.Array):
        def forward(x):
            fx, jx = vmap_iteration_step(x, *tuples.T)
            return jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)

        return broydenb2(forward, x0)

    tuples = cartesian_product(mu, k, V, t)
    x0 = jnp.ones((tuples.shape[0], delta_sites.shape[0]), dtype=jnp.complex128)

    return solve(tuples, x0)
    assert False

    # # Get all the tuples of mu, k, V and t we are considering

    # # TODO: Add MPI here to divide these tuples
    # # TODO: Limit memory usage if too many requests

    # @jax.jit
    # def cond_func(state):
    #     i, x, done = state
    #     return (i < max_iter) & ~jnp.all(done)

    # def body_func(state):
    #     i, x, done = state

    #     mask = ~done

    #     fx, Jx = forward(x, mask)

    #     norm = jnp.linalg.norm(fx, axis=-1)
    #     new_done = norm < eps
    #     print(jnp.mean(norm), jnp.mean(done), jnp.sum(done))

    #     done = done.at[mask].set(new_done)

    #     dx = - jnp.linalg.solve(Jx, fx[..., None]).squeeze(-1)  # of size mask

    #     x_next = x.at[mask].add(dx)
    #     # x_next = fx
    #     return i + 1, x_next, done

    # done0 = jnp.zeros(tuples.shape[0], dtype=bool)
    # init_val = (
    #     0,
    #     jnp.ones((done0.size, delta_indices.size), dtype=jnp.complex128),
    #     done0,
    # )
    # val = init_val
    # while cond_func(val):
    #     val = body_func(val)

    # return val


def main():
    sys = CubicLattice((100, 1, 1), (True, False, False))

    N = 5
    num_temp = 5
    x, aux = order_parameters(
        sys,
        mu=jnp.array([0.05]),
        k=jnp.pi * (2 * jnp.arange(N) + 1) / (2 * N),
        V=jnp.array([0.7]),
        # t =jnp.array([0.0])
        t=jnp.linspace(0.0, 0.01, num_temp),
    )
    print(f"Number of iterations: {aux.iterations}")
    matr = x.reshape((N, num_temp, -1)).mean(0)

    import matplotlib.pyplot as plt

    for i in range(num_temp):
        plt.plot(matr[i, :], label=f"{i}")
    plt.legend()
    plt.savefig("temp.pdf")

    print(matr)
    print(matr.shape)


if __name__ == "__main__":
    main()
