import torch
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
from gradient import batch_consistency, eigenvalue_perturbation_gradient
from typing import Optional


def find_blocks(matrix: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Takes in a matrix an either returns it as is or returns its block structure


    Args:
        matrix (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    _, N, N = matrix.shape

    mask = torch.where(matrix[0].abs() > 0, 1, 0).numpy()
    csgraph = sp.coo_array(mask)

    num_blocks, labels = connected_components(csgraph=csgraph, directed=False)
    if num_blocks == 1:
        # Default to regular in the case of no blocks
        return matrix.unsqueeze(1), []

    indices = torch.arange(N)
    blocks = [indices[labels == i] for i in range(num_blocks)]

    new_matrix = torch.stack(
        [matrix[:, blk.unsqueeze(-1), blk.unsqueeze(-2)] for blk in blocks], dim=1
    ).to(matrix.dtype)
    return new_matrix, blocks


def block_diagonalize(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Takes in a tensor and returns
            L, Q = torch.linalg.eigh(matrix)
        only it checks first whether the matrix is actually block-diagonal

    Args:
        matrix (torch.Tensor): _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_
    """
    B, N, N = matrix.shape

    matrix, blocks = find_blocks(matrix)

    B, num_blocks, _, _ = matrix.shape

    print("Performing diagonalization")
    L, Q = torch.linalg.eigh(matrix)

    if num_blocks == 1:
        return L.squeeze(1), Q.squeeze(1)

    print("Gathering blocks")
    res_L = torch.zeros(B, N).to(L.dtype)
    res_Q = torch.zeros(B, N, N).to(Q.dtype)

    for n in range(matrix.shape[1]):
        blk = blocks[n]
        res_L[:, blk] = L[:, n, :]
        res_Q[:, blk.unsqueeze(-1), blk.unsqueeze(-2)] = Q[:, n, :, :]

    print("Sorting the eigenvalues")
    # Sort L in ascending order
    sorted_indices = torch.argsort(res_L, dim=1)
    res_L = torch.gather(res_L, 1, sorted_indices)
    # Sort the eigenvectors, which are located at the colums
    res_Q = torch.gather(res_Q, -1, sorted_indices.unsqueeze(1).expand(-1, N, -1))

    # print(res_L.shape, res_Q.shape)
    # assert(False)
    return res_L, res_Q


@torch.jit.script
def insert_deltas(
    deltas: torch.Tensor,
    base_matrix: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
):
    # x is of shape (batch, N) or (1, N) if it is to be broadcasted to batch.
    # base_matrix is of shape (batch, N, N)
    B, _ = deltas.shape
    jsigma2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)

    top_vals = deltas.view(B, -1, 1, 1) * jsigma2  # (batch, N, 2, 2)

    bot_vals = top_vals.conj().transpose(-2, -1)

    # print(row.shape, col.shape, base_matrix.shape )
    base_matrix[:, row.unsqueeze(-1), col.unsqueeze(-2)] = top_vals
    base_matrix[:, col.unsqueeze(-1), row.unsqueeze(-2)] = bot_vals

    # Returns base_matrix of shape (B, N, N)
    return base_matrix


class BDGFunction(torch.nn.Module):
    """Calculates f(deltas) and J(deltas) for batches in parallel."""

    def __init__(
        self,
        base_matrix: torch.Tensor,
        beta: torch.Tensor,
        idx: torch.Tensor,
        pot: torch.Tensor,
    ):
        super().__init__()
        self.base_matrix = base_matrix
        self.beta = beta

        self.idx = idx
        self.pot = pot
        self.row = 4 * self.idx[:, None] + torch.arange(2)
        self.col = 4 * self.idx[:, None] + torch.arange(2) + 2

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask_ = torch.arange(self.base_matrix.size(0), dtype=torch.long)
            x = x.expand(self.base_matrix.size(0), -1)
        else:
            mask_ = mask
        L, Q = torch.linalg.eigh(
            insert_deltas(x, self.base_matrix[mask_], self.row, self.col)
        )

        f_future = torch.jit.fork(
            batch_consistency, L, Q, self.beta, self.idx, self.pot
        )
        J_future = torch.jit.fork(
            eigenvalue_perturbation_gradient, L, Q, self.beta, self.idx, self.pot
        )

        f = torch.jit.wait(f_future)
        J = torch.jit.wait(J_future)

        return f - x, J - torch.eye(J.shape[-1], dtype=J.dtype)
