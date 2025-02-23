import jax
import jax.numpy as jnp

@jax.jit
def insert_blocks(mat: jnp.ndarray, block_indices: jnp.ndarray, blocks: jnp.ndarray):
    """ Returns mat with blocks inserted at block indices

    Args:
        mat (jnp.ndarray): 2N by 2N
        block_indices (jnp.ndarray): (num_blocks, 2). (blk_row, blk_col) x num_blocks
        blocks (jnp.ndarray): (num_blocks, 2, 2)

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