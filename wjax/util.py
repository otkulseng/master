import jax
import jax.numpy as jnp

@jax.jit
def insert_blocks(mat: jax.Array, block_indices: jax.Array, blocks: jax.Array):
    """ Returns mat with blocks inserted at block indices

    Args:
        mat (jax.Array): 2N by 2N
        block_indices (jax.Array): (num_blocks, 2). (blk_row, blk_col) x num_blocks
        blocks (jax.Array): (num_blocks, 2, 2)

    Returns:
        _type_: 2N by 2N with blocks updated
    """
    orig_pos = 2 * block_indices
    num_blocks = block_indices.shape[0]

    row_offset = jnp.array([0, 0, 1, 1])
    col_offset = jnp.array([0, 1, 0, 1])

    rows = orig_pos[:, 0].repeat(row_offset.size) + jnp.tile(row_offset, num_blocks)
    cols = orig_pos[:, 1].repeat(col_offset.size) + jnp.tile(col_offset, num_blocks)
    return mat.at[rows, cols].add(blocks.flatten())

@jax.jit
def add_diagonal(mat: jax.Array, kvals: jnp.float32, mask: jax.Array):
    """ Takes in a matrix (N, N) and number kvals and returns a tensor of
    shape (N, N) with kvals * mask added to the diagonal

    Args:
        mat (jax.Array): N x N square matrix
        kvals (jax.Array): B. Array of numbers
        mask (jax.Array): N. Diagonal mask
    """

    x, y = jnp.diag_indices(mat.shape[-1])
    return mat.at[x, y].add(kvals * mask)

# @jax.jit
def generate_chebyshev_kmodes(N: int):
    return jnp.cos(jnp.pi * (jnp.arange(N) * 2 + 1) / (2 * N))
