import torch
import numpy as np

class BlockSparseMatrix:
    def __init__(self, indices, blocks):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        if not torch.is_tensor(blocks):
            blocks = torch.tensor(blocks)

        indices = indices.to(torch.long)
        blocks = blocks.to(torch.complex128)

        self.indices = indices
        self.row_indices = indices[:, 0]
        self.col_indices = indices[:, 1]
        self.block_indices = indices[:, 2]
        self.blocks = blocks

        self.size = int((self.row_indices.max().item() + 1) * self.blocks[0].shape[0])

    def __floordiv__(self, other):
        self.blocks /= other
        return self

    def __truediv__(self, other):
        self.blocks /= other
        return self

    def norm(self):
        # Return a matrix norm
        temp = BlockSparseMatrix(self.indices, torch.abs(self.blocks))
        return temp.matvec(
            torch.ones((1, self.size, 2), dtype=self.blocks.dtype)
        ).abs().max().item()

    def to_dense(self):
        row_size = int(self.row_indices.max().item() + 1)
        col_size = int(self.col_indices.max().item() + 1)
        row_shape, col_shape = self.blocks[0].shape
        out = torch.zeros(
            size=(row_size * row_shape, col_size*col_shape),
        ).to(self.blocks.dtype)

        for i, j, k in self.indices:
            out[i*row_shape:(i+1)*row_shape ,j*col_shape:(j+1)*col_shape] = self.blocks[k]
        return out

    def matvec(self, V: torch.Tensor):
        # Assume V is of shape (num_batches, N, 2) where N is the max number of
        # columns and 2 is the column size.
        # Let nnz = Number of NonZero
        out = torch.zeros_like(V)
        num_batches = V.shape[0]

        V = V[:, self.col_indices, :].unsqueeze(-1) # (num_batches, nnz, 2, 1)

        blocks = self.blocks[self.block_indices].unsqueeze(0) # (1, nnz, 2, 2)


        result_blocks = torch.matmul(blocks, V).squeeze(-1) # (num_batches, nnz, 2)

        # Row indices expanded from nnz to (num_batches, nnz, 2)
        indices_expanded = self.row_indices.unsqueeze(0).unsqueeze(-1).expand(num_batches, -1, 2)
        # print(indices_expanded.shape)
        # assert(False)

        out.scatter_add_(dim=1, index=indices_expanded, src=result_blocks) # (num_batches, N, 2)
        return out


import matplotlib.pyplot as plt
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

    shape = (50, 2, 2)
    vec = torch.randn(size=shape).to(torch.float32)

    res = torch.einsum("...ij, ...ij -> ...", vec.conj(), matr.matvec(vec))


    print(torch.mean(res))




if __name__ == "__main__":
    main()