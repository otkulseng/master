import jax
import jax.numpy as jnp
from util import insert_blocks


@jax.jit
def make_bdg_H_term(idx: jax.Array, blk: jax.Array):
    row = idx[:, 0]
    col = idx[:, 1]
    new_row = jnp.concat([2 * row, 2 * row + 1], axis=0)
    new_col = jnp.concat([2 * col, 2 * col + 1], axis=0)
    new_blk = jnp.concat([blk, -jnp.conj(blk)], axis=0)
    return jnp.stack([new_row, new_col], axis=1), new_blk


@jax.jit
def make_bdg_D_term(idx: jax.Array, blk: jax.Array):
    row = idx[:, 0]
    col = idx[:, 1]
    new_row = jnp.concat([2 * row, 2 * col + 1], axis=0)
    new_col = jnp.concat([2 * col + 1, 2 * row], axis=0)
    new_blk = jnp.concat([blk, jnp.transpose(jnp.conj(blk), axes=(0, 2, 1))])
    return jnp.stack([new_row, new_col], axis=1), new_blk


# TODO: Finish this. Implementing the gradient here is key to obtaining the jacobian
# @jax.jit
# def LQ(sys: BDGMatrix, D: jax.Array):
#     pass


class BDGMatrix:
    def __init__(self, H_idx, H_blk, V_idx, V_blk, N):
        # Convert to static types, such that jit-functions can accept
        # staticBDGMatrix.
        self.H_idx = jnp.array(H_idx)
        self.H_blk = jnp.array(H_blk)
        self.V_idx = jnp.array(V_idx)
        self.V_blk = jnp.array(V_blk)
        self.N = int(N)

        self.pot_idx = self.V_idx[:, 0]

    def tree_flatten(self):
        # Return no children; treat all fields as static auxiliary data.
        return (), (self.H_idx, self.H_blk, self.V_idx, self.V_blk, self.N)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


jax.tree_util.register_pytree_node(
    BDGMatrix, BDGMatrix.tree_flatten, BDGMatrix.tree_unflatten
)

# 24 tråder
# 8
# 3 processer

# multiproccessing
# MPI: Message Pass Interface


@jax.jit
def to_dense(sys: BDGMatrix, D: jax.Array):
    mat = jnp.empty((4 * sys.N, 4 * sys.N), dtype=jnp.complex64)

    H_idx, H_blk = make_bdg_H_term(sys.H_idx, sys.H_blk)

    # TODO: This would change as well when v_blk are blocks
    jsigma2 = jnp.array([[0, 1], [-1, 0]], dtype=jnp.complex64)
    v_blk_temp = D[:, None, None] * jsigma2[None, :, :]  # shape (nnz, 2, 2)
    V_idx, V_blk = make_bdg_D_term(sys.V_idx, v_blk_temp)
    idx = jnp.concatenate([H_idx, V_idx], axis=0)
    blk = jnp.concatenate([H_blk, V_blk], axis=0)
    return insert_blocks(mat, idx, blk)


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


@jax.jit
def self_consistency_equation(sys: BDGMatrix, D: jax.Array, kmode: float, beta: float):
    # 4N by 4N matrix
    matr: jax.Array = to_dense(sys, D)
    size = matr.shape[0]
    N = size // 4

    # Add kmode to diagonal
    mask = jnp.tile(jnp.array([1, 1, -1, -1], dtype=jnp.complex64), N)
    x, y = jnp.diag_indices(size)
    matr = matr.at[x, y].add(kmode * mask)

    # Do the eigenvalue decomposition
    L, Q = jnp.linalg.eigh(matr)
    # L, Q = my_eigh(matr)

    # Perform the self-consistency
    corr = temperature_independent_correlations(Q) # (Eigenvalue, Position) : (4N, N)
    tanh = tanhify_eigenvalues(L, beta) # (Eigenvalue, ) : (4N,)
    Vpot = sys.V_blk # (Position)

    res = tanh[:, None] * corr[:, sys.pot_idx] * Vpot[None, :] # (Eigenvalues, nnz)

    # Keep only positive eigenvalues
    res = res[2*N:, :]

    # Sum over eigenvalues
    return jnp.sum(res, axis=0) # (nnz)