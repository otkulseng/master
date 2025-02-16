import torch


@torch.jit.script
def batch_consistency(
    L: torch.Tensor,
    Q: torch.Tensor,
    Idx: torch.Tensor,
    Beta: torch.Tensor,
    BatchWeight: torch.Tensor,
    Potential: torch.Tensor
):
    L = torch.real(L)

    B = L.shape[0]
    size = L.shape[1]
    N = size // 4

    Q = Q.to(torch.complex128)
    Q = Q.transpose(-2, -1).view(L.shape[0], L.shape[1], -1, 4)  # (B, 4N, N, 4)

    # Keep only positive eigenvalues
    mask = L > 0
    Q = Q[mask].view(B, 2 * N, N, 4)

    # L: (num_batches, 2N) positive eigenvalues
    L = L[mask].view(Q.shape[0], -1)

    # Q: (num_batches, 2N, N, 4) == (num_batches, eigenvalues, position, Nambu)
    Q = Q[:, :, Idx, :]  # (B, 2N, nnz, 4)

    udo: torch.Tensor = Q[:, :, :, 1]
    vup: torch.Tensor = Q[:, :, :, 2]
    tanhe = torch.tanh(Beta * L / 2)

    return (BatchWeight @ torch.sum(
        (udo.conj() * vup) * Potential.view(1, 1, -1) * tanhe.unsqueeze(-1),
        dim=1,
    ))


@torch.jit.script
def eigenvalue_perturbation_gradient(
    L: torch.Tensor,
    Q: torch.Tensor,
    Idx: torch.Tensor,
    Beta: torch.Tensor,
    BatchWeight: torch.Tensor,
    Potential: torch.Tensor
):
    """_summary_

    Args:
        L (torch.Tensor): _description_
        Q (torch.Tensor): _description_
        Idx (torch.Tensor): _description_
    """

    eps = 1e-8
    K = eps * torch.tensor(
        [
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.complex128,
    )
    B, size = L.shape
    N = size // 4

    Q_b = Q.transpose(-1, -2).view(B, 4 * N, N, 4).transpose(1, 2)
    Q_b = Q_b[:, Idx, :, :]

    # Batched multiplication
    K_Q = torch.matmul(Q_b, K.view(1, 1, 4, 4))

    Q_b = Q_b.transpose(-2, -1)

    denom = L.unsqueeze(-2) - L.unsqueeze(-1)  # (B, 4N, 4N)
    denom = torch.where(
        denom.abs() < 1e-10, torch.inf, denom
    )  # Zeros out contributions in this case

    out = torch.zeros((N, N), dtype=torch.complex128)

    # x0 = batch_consistency(L, Q, Idx, Beta, BatchWeight, Potential)
    for n in range(K_Q.size(1)):
        # Numerator in eigenvalue perturbation
        Q_K_Q = torch.matmul(K_Q[:, n, :, :], Q_b[:, n, :, :])  # (B, 4N, 4N)

        # Change in eigenvalues on the diagonal
        L_diff = torch.diagonal(Q_K_Q, dim1=-2, dim2=-1)  # (B, 4N)

        # Eigenvector factors are Q_K_Q / denom
        evec_factor = Q_K_Q.transpose(-2, -1) / denom  # (B, 4N, 4N)

        Q_diff = torch.matmul(Q, evec_factor)  # (B, 4N, 4N)

        # dx = batch_consistency(L + L_diff, Q + Q_diff, Idx, Beta)
        dx = batch_consistency(
            L + L_diff,
            Q + Q_diff,
            Idx,
            Beta,
            BatchWeight,
            Potential
        ) 
        out[:, n] = dx / eps
    return out
