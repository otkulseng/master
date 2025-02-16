import torch

@torch.jit.script
def batch_consistency(
    L: torch.Tensor,
    Q: torch.Tensor,
    Idx: torch.Tensor,
    Beta: torch.Tensor,
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

    return torch.sum(
        (udo.conj() * vup) * Potential.view(1, 1, -1) * tanhe.unsqueeze(-1),
        dim=1,
    )

# @torch.jit.script
def eigenvalue_perturbation_gradient(
    L: torch.Tensor,
    Q: torch.Tensor,
    Idx: torch.Tensor,
    Beta: torch.Tensor,
    Potential: torch.Tensor,
):
    """_summary_

    Args:
        L (torch.Tensor): _description_
        Q (torch.Tensor): _description_
        Idx (torch.Tensor): _description_
    """

    K =  torch.tensor(
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
    nnz = Potential.shape[0]


    Q_b = Q.transpose(-1, -2).view(B, 4 * N, N, 4).transpose(1, 2)
    Q_b = Q_b[:, Idx, :, :]

    # Batched multiplication
    K_Q = torch.matmul(Q_b[:, :, -2*N:, :], K.view(1, 1, 4, 4))

    Q_b = Q_b.transpose(-2, -1)

    denom = L.unsqueeze(-1) - L.unsqueeze(-2)  # (B, 4N, 4N)
    denom = torch.where(
        denom.abs() < 1e-10, torch.inf, denom
    )[:, -2*N:, :]  # Zeros out contributions in this case


    # Keep positive eigenvalues
    tanhe = torch.tanh(Beta * L[:, -2*N:] / 2) # (B, 2N)

    Q = Q.transpose(-1, -2).view(B, 4*N, N, 4)
    Q = Q[:, :, Idx, 1:3]

    udo_0 = Q[:, -2*N:, :, 0]
    vup_0 = Q[:, -2*N:, :, 1]

    out = torch.zeros((B, nnz, nnz), dtype=torch.complex128)
    for n in range(nnz):
        # Numerator in eigenvalue perturbation
        Q_K_Q = torch.matmul(K_Q[:, n, :, :], Q_b[:, n, :, :])  # (B, 4N, 4N)
        evec_factor = Q_K_Q / denom  # (B, 4N, 4N)
        Q_diff = torch.einsum("bij, bjkl->bikl", evec_factor, Q)

        udiff = Q_diff[..., 0]
        vdiff = Q_diff[..., 1]

        out[:, :, n] = torch.sum(
            (udo_0.conj() * vdiff + udiff.conj() * vup_0) * Potential.view(1, 1, -1) * tanhe.unsqueeze(-1),
            dim=1,
        )

    return out
