import torch
import numpy as np

class BlockSparseMatrix:
    def __init__(self, indices, blocks):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        if not torch.is_tensor(blocks):
            blocks = torch.tensor(blocks, dtype=torch.complex128)

        self.indices = indices
        self.row_indices = indices[:, 0]
        self.col_indices = indices[:, 1]
        self.block_indices = indices[:, 2]

        self.blocks = blocks
        self.num_block_rows = int(self.block_indices.max().item()) + 1

    def matvec(self, V: torch.Tensor):
        # Assume V is of shape (num_batches, N, 2) where N is the max number of
        # columns and 2 is the column size.
        # Let nnz = Number of NonZero
        num_batches = V.shape[0]

        V = V[:, self.col_indices, :].unsqueeze(-1) # (num_batches, nnz, 2, 1)
        blocks = self.blocks[self.block_indices].unsqueeze(0) # (1, nnz, 2, 2)
        result_blocks = torch.matmul(blocks, V).squeeze(-1) # (num_batches, nnz, 2)
        out = torch.zeros((num_batches, self.num_block_rows, 2), dtype=result_blocks.dtype)

        # Row indices expanded from nnz to (num_batches, nnz, 2)
        indices_expanded = self.row_indices.unsqueeze(0).unsqueeze(-1).expand(num_batches, -1, 2)

        out.scatter_add_(dim=1, index=indices_expanded, src=result_blocks) # (num_batches, N, 2)
        return out


def main():
    matr = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ]

    blocks = [
        [
            [1, 2],
            [3, 4],
        ], [
            [5, 6],
            [7, 8]
        ]
    ]
    matr = BlockSparseMatrix(matr, blocks)

    shape = (1000000, 2, 2)
    vec = torch.randn(size=shape).to(torch.complex128)

    res = torch.einsum("...ij, ...ij -> ...", vec, matr.matvec(vec))

    print(torch.mean(res))



if __name__ == "__main__":
    main()