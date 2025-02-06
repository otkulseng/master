from matmul import BlockSparseMatrix
import torch
import numpy as np


def bdg_base_matrix(indices: torch.Tensor, blocks: torch.Tensor):
    # Creates the H_ij terms of the BDG matrix. That is,
    # from H_ij, at the indices and blocks specified, get
    # [[H_ij, 0],]
    # [[0, -H_ij^*],]
    # With the indices doubled
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long)

    if not torch.is_tensor(blocks):
        blocks = torch.tensor(blocks, dtype=torch.float32)
    row = indices[:, 0]
    col = indices[:, 1]
    blk = indices[:, 2]

    num_blocks = int(blk.max().item()) + 1

    row = torch.concat([2*row, 2*row + 1])
    col = torch.concat([2*col, 2*col + 1])
    blk = torch.concat([blk, blk + num_blocks])
    blocks = torch.vstack(
        [blocks, -blocks.conj()]
    )

    indices = torch.vstack([row, col, blk]).T

    return BlockSparseMatrix(indices, blocks)

def bdg_order_matrix_indices(indices: torch.Tensor):
    row = indices[:, 0]
    col = indices[:, 1]

    row = torch.concat([2*row, 2*row + 1])
    col = torch.concat([2*col + 1, 2*col])

    # First regular, then conjugated
    return torch.vstack([row, col]).T

def apply_order_parameters(indices: torch.Tensor, order_parameters: torch.Tensor, V: torch.Tensor):
    # V is of shape (num_batches, N, 2)
    # Where N is the block shape. Indices can be maximally N.
    # For s-wave, the multiplication is always with jsigma2 = [[0, 1], [-1, 0]] which
    # represent a simple shift

    # order parameters must be order_parameters = torch.concat([deltas, -deltas.conj()])

    # Instead of matrix multiplication, just permute the rows

    row = indices[:, 0]
    col = indices[:, 1]

    res = torch.zeros_like(V)

    blk = torch.tensor(
        [[0, 1],
         [-1, 0]], dtype=torch.float32
    ) # (2, 2)

    # (num_batch, nnz, 2) * (1, nnz, 1)
    res[:, row, :] = torch.matmul(V[:, col, :], blk.T) * order_parameters.unsqueeze(0).unsqueeze(-1)
    return res

def main():
    indices = torch.tensor([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=torch.long)

    order_params = torch.tensor([1])
    order_params = torch.concat([order_params, -order_params.conj()])

    indices = torch.tensor([
        [1, 1],
        # [1, 1]
    ])

    rowcol = bdg_order_matrix_indices(indices)


    shape = (1, 4, 2)
    vec = torch.randn(size=shape).to(torch.float32)
    res = apply_order_parameters(rowcol, order_params, vec)

    print(res)
    assert(False)



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