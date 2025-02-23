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

def rel_diff(a, b):
    return torch.abs(a - b) /(1e-15 + torch.abs(a + b))

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

    @torch.jit.export
    def LQ(self, x: torch.Tensor):
        return torch.linalg.eigh(insert_deltas(x, self.base_matrix, self.row, self.col))

    @torch.jit.export
    def matrix(self, x: torch.Tensor):
        return insert_deltas(x, self.base_matrix, self.row, self.col)

    def get_rho(self, L, Q, beta, weights):
        M = torch.einsum(
            "b, bij->ij",
            weights,
            eigenvalue_perturbation_gradient(L, Q, beta, self.idx, self.pot),
        )

        return torch.linalg.norm(M, ord=2)

    @torch.jit.export
    def critical_temperature(
        self,
        weights: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        Q: Optional[torch.Tensor] = None,
        eps: torch.Tensor = torch.tensor(1e-4),
        max_iter: int = 100,
        min_temp: Optional[torch.Tensor] = None,
        max_temp: Optional[torch.Tensor] = None
    ):

        if L is None or Q is None:
            L, Q = self.LQ(torch.zeros(1, self.pot.size(0), dtype=torch.complex128))

        if min_temp is None:
            min_temp = torch.tensor(0.0) +1e-15
        if max_temp is None:
            max_temp = torch.tensor(1.0)

        min_rho = torch.jit.fork(self.get_rho, L, Q, 1 / min_temp, weights)
        max_rho = torch.jit.fork(self.get_rho, L, Q, 1 / max_temp, weights)

        min_rho = torch.jit.wait(min_rho)
        max_rho = torch.jit.wait(max_rho)

        # Ensure min-rho and max-rho are on different sides of 1
        if not (min_rho > 1 and max_rho < 1):
            return min_temp


        for it in range(max_iter):
            curr = rel_diff(min_temp, max_temp)
            if curr < eps:
                break

            lower = (2 * min_temp + max_temp) / 3
            higher = (min_temp + 2*max_temp) / 3
            lower_rho = torch.jit.fork(self.get_rho, L, Q, 1 / lower, weights)
            higher_rho = torch.jit.fork(self.get_rho, L, Q, 1 / higher, weights)

            lower_rho = torch.jit.wait(lower_rho)
            higher_rho = torch.jit.wait(higher_rho)


            # Have min_temp, lower_temp, higher_temp, max_temp
            #
            if lower_rho < 1:
                max_temp = lower
            elif lower_rho > 1 and higher_rho < 1:
                min_temp = lower
                max_temp = higher
            else:
                min_temp = higher

            print(f"{it}: rho={lower_rho.item(), higher_rho.item()}\t temp={min_temp.item(),max_temp.item()}\t eps={curr.item()}")

        # Do interpolation search. Assume rho changes linearly betwewn minval and maxval

        return (max_temp + min_temp) / 2


def rho_based_critical_temperature(matr_func: BDGFunction, weights: torch.Tensor, min_temp = 0.0, max_temp=1.0):
    nnz = matr_func.pot.size(0)  # nnz. I.e. number of order parameters
    matr: torch.Tensor = matr_func.matrix(torch.zeros(1, nnz, dtype=torch.complex128))

    # TODO: Make torchscript-compatible blocks diagonalization to make this function redundant
    L, Q = block_diagonalize(matr)

    return matr_func.critical_temperature(weights, L, Q, min_temp=torch.tensor(min_temp), max_temp=torch.tensor(max_temp))

