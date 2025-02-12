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
        blocks = torch.tensor(blocks, dtype=torch.complex128)
    row = indices[:, 0]
    col = indices[:, 1]
    blk = indices[:, 2]

    num_blocks = blocks.size(0)


    row = torch.concat([2*row, 2*row + 1])
    col = torch.concat([2*col, 2*col + 1])
    blk = torch.concat([blk, blk + num_blocks])
    blocks = torch.vstack(
        [blocks, -blocks.conj()]
    )

    indices = torch.vstack([row, col, blk]).T

    return indices, blocks

def bdg_potential_matrix(indices: torch.Tensor, blocks: torch.Tensor):
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long)

    if not torch.is_tensor(blocks):
        blocks = torch.tensor(blocks, dtype=torch.complex128)
    row = indices[:, 0]
    col = indices[:, 1]
    blk = indices[:, 2]

    num_blocks = blocks.size(0)


    row = torch.concat([2*row, 2*row + 1])
    col = torch.concat([2*col + 1, 2*col])
    blk = torch.concat([blk, blk + num_blocks])

    blocks = torch.vstack(
        [blocks, blocks.T.conj()]
    )

    indices = torch.vstack([row, col, blk]).T

    return indices, blocks

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
        [[0, -1],
         [1, 0]], dtype=torch.complex128
    ) # (2, 2)

    # (num_batch, nnz, 2) * (1, nnz, 1)
    res[:, row, :] = torch.matmul(V[:, col, :], blk.T) * order_parameters.unsqueeze(0).unsqueeze(-1)
    return res


def insert(base_matrix: torch.Tensor, indices: torch.Tensor, blocks: torch.Tensor):
    row_shape, col_shape = blocks[0].shape

    for i, j, k in indices:
        base_matrix[i*row_shape:(i+1)*row_shape ,j*col_shape:(j+1)*col_shape] = blocks[k]
    return base_matrix

class SelfConsistentBDGSolver:
    def __init__(self, H_matrix: torch.Tensor,
                        V_indices: torch.Tensor, V_blocks: torch.Tensor,
                        temperature: torch.Tensor):

        self.H_matrix = H_matrix
        self.V_indices, self.V_blocks = bdg_potential_matrix(V_indices, V_blocks)

        self.potential_indices = V_indices

        self.beta = 1.0 / (1e-15 + temperature)

    def from_kmode(self, kmode: torch.Tensor):
        N = self.H_matrix.size(0) // 4
        diag_pattern = torch.tensor([1, 1, -1, -1], dtype=self.H_matrix.dtype).repeat(N)
        return SelfConsistentBDGSolver(
            H_matrix= self.H_matrix.diagonal().add(kmode * diag_pattern),
            V_indices=self.V_indices,
            V_blocks=self.V_blocks
        )

    def build_matrix(self, x: torch.Tensor):
        # Insert the order parameters
        x = torch.concat([x, x.conj()]).unsqueeze(-1).unsqueeze(-1)
        return insert(self.H_matrix, self.V_indices, x * self.V_blocks)

    def diagonalize(self, matrix: torch.Tensor):
        eigvals, eigvecs = torch.linalg.eigh(matrix)

        # structure eigvecs[Eigenvalue, Position, Particle]
        eigvecs = torch.reshape(eigvecs.T, (eigvals.size, -1, 4))
        return eigvals, eigvecs

    def corr(self, eigvals: torch.Tensor, eigvecs: torch.Tensor, beta: torch.Tensor):
        uup = eigvecs[:, :, 0]
        udo = eigvecs[:, :, 1]
        vup = eigvecs[:, :, 2]
        vdo = eigvecs[:, :, 3]

        return torch.einsum("ni, n->i",
                             udo.conj()*vup - uup.conj()*vdo,
                             torch.tanh(beta * eigvals / 2)) / 2

    def forward(self, x: torch.Tensor):
        matrix = self.build_matrix(x)
        evals, evecs =  self.diagonalize(matrix)
        corr = self.corr(evals, evecs, beta=self.beta)
        



















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


def main():

    test = torch.arange(4)
    matr = torch.zeros((4, 4))

    matr[0, :] = test
    print(matr)

    test[0] = 2
    print(test)
    print(matr)

if __name__ == "__main__":
    main()