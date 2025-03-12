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
        # BDGMatrix.
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
def matrix(sys: BDGMatrix, D: jax.Array, kmode: float):
    # Create the matrix

    # 4N by 4N matrix
    matr: jax.Array = to_dense(sys, D)
    size = matr.shape[0]
    N = size // 4
    # Add kmode to diagonal
    mask = jnp.tile(jnp.array([1, 1, -1, -1], dtype=jnp.complex64), N)
    x, y = jnp.diag_indices(size)
    matr = matr.at[x, y].add(kmode * mask)
    return matr





@jax.jit
def self_consistency_equation(sys: BDGMatrix, D: jax.Array, kmode: float, beta: float):
    # L, Q = my_eigh(matr)
    L, Q = jnp.linalg.eigh(matrix(sys, D, kmode))
    return consistency(sys, L, Q, beta)


def perturb(idx, L, Q):
    # Q has shape (4N, N, 4)

    Qtilde: jax.Array = Q[:, idx, :]  # (4N, 4)

    pert_matr = jnp.array(
        [
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=Qtilde.dtype,
    )

    numerator = (Qtilde @ pert_matr) @ jnp.transpose(Qtilde.conj(), (1, 0))  # (4N, 4N)

    Ldiff = jnp.diag(Qtilde)  # (4N, )

    # L has shape (4N, )
    denom = L[None, :] - L[:, None]  # (4N, 4N)
    denom = jnp.where(
        jnp.abs(denom) < 1e-10, jnp.inf, denom
    )  # Zero out the contributions that are too large.

    frac = numerator / denom  # (4N, 4N)

    Qdiff = jnp.tensordot(
        frac, Q, axes=([1], [0])
    )  # multiply last dimension of frac with first dimension of Q. Shape (4N, N, 4)

    return Ldiff, Qdiff


def single_index_jacobian(sys: BDGMatrix, idx, beta, L, Q):
    _, Qdiff = perturb(idx, L, Q)

    Q = Q[2*sys.N:,sys.pot_idx, :] # (2N, nnz, 4)
    L = L[2*sys.N:] # (2N, )
    Qdiff = Qdiff[2*sys.N:, sys.pot_idx, :] # (2N, nnz, 4)

    tanhe = tanhify_eigenvalues(L, beta)

    uup0: jax.Array = Q[..., 0]
    udo0: jax.Array = Q[..., 1]
    vup0: jax.Array = Q[..., 2]
    vdo0: jax.Array = Q[..., 3]

    # Qdiff of shape (4N, N, 4)
    uup_diff: jax.Array = Qdiff[..., 0]
    udo_diff: jax.Array = Qdiff[..., 1]
    vup_diff: jax.Array = Qdiff[..., 2]
    vdo_diff: jax.Array = Qdiff[..., 3]

    res = (
        tanhe[:, None]
        * (
            (udo0.conj() * vup_diff + udo_diff.conj() * vup0)
            - (uup0.conj() * vdo_diff + uup_diff.conj() * vdo0)
        )
        / 2
    )  # (4N, N)

    return jnp.sum(res, axis=0) * sys.V_blk  # (nnz,)


def jacobian(sys: BDGMatrix, L: jax.Array, Q: jax.Array, beta: float):
    # Cannot vmap entire jacobian, as this requires nnz * N^2 memory.
    # Scan instead (which, at any point, only requires N^2 memory)

    # Now, Q[n, :] is eigenvector corresponding to eigenvalue E_n
    Q = jnp.swapaxes(Q, -1, -2)

    # (Eigenvalue, Position, Nambu)
    Q = jnp.reshape(Q, shape=(Q.shape[-2], sys.N, 4))

    return jax.vmap(single_index_jacobian, in_axes=(None, 0, None, None, None))(
        sys, sys.pot_idx, beta, L, Q
    )


@jax.jit
def consistency_val_and_jac(sys: BDGMatrix, D: jax.Array, kmode: float, beta: float):
    # Share matrix creation and diagonalisation.
    L, Q = jnp.linalg.eigh(matrix(sys, D, kmode))
    return consistency(sys, L, Q, beta) - D, -jacobian(sys, L, Q, beta) - jnp.eye(D.shape[0], dtype=D.dtype)
