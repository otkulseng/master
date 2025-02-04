from matmul import BlockSparseMatrix
import torch
import numpy as np


def bdg_base_matrix(indices: torch.Tensor, blocks: torch.Tensor):
    # Creates the H_ij terms of the BDG matrix. That is,
    # from H_ij, at the indices and blocks specified, get
    # [[H_ij, 0],]
    # [[0, -H_ij^*],]
    # With the indices doubled
    row = indices[:, 0]
    col = indices[:, 1]
    blk = indices[:, 2]

    num_blocks = int(blk.max().item()) + 1

    row = torch.concat([2*row, 2*row + 1])
    col = torch.concat([2*col, 2*col + 1])
    blk = torch.concat([blk, blk + num_blocks])
    blocks = torch.vstack(
        [blocks, -np.conj(blocks)]
    )

    indices = torch.vstack([row, col, blk]).T

    return BlockSparseMatrix(indices, blocks)

def bdg_order_matrix_indices(indices: torch.Tensor):
    row = torch.concat([2*row, 2*row + 1])
    col = torch.concat([2*col + 1, 2*col])

    # First regular, then conjugated
    return torch.vstack([row, col])

def apply_order_parameters(indices: torch.Tensor, order_parameters: torch.Tensor, V: torch.Tensor):
    # V is of shape (num_batches, N, 2)
    # Where N is the block shape. Indices can be maximally N.
    # For s-wave, the multiplication is always with jsigma2 = [[0, 1], [-1, 0]] which
    # represent a simple shift

    # Instead of matrix multiplication, just perturb the rows
    row = indices[:, 0]
    col = indices[:, 1]
    V = V[:, col, ]
    Vmult = torch.stack([V[..., 1], -V[..., 0]], dim=-1)



def main():
    indices = torch.tensor([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=torch.long)

    blocks = torch.tensor([
        [
            [1, 2],
            [3, 4],
        ], [
            [5, 6],
            [7, 8]
        ]
    ])
    bdg_base_matrix(indices, blocks)

if __name__ == "__main__":
    main()