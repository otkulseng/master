import torch


@torch.jit.script
def batch_consistency(
    L: torch.Tensor,
    Q: torch.Tensor,
    Beta: torch.Tensor,
    Idx: torch.Tensor,
    Potential: torch.Tensor,
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
    uup: torch.Tensor = Q[:, :, :, 0]
    udo: torch.Tensor = Q[:, :, :, 1]
    vup: torch.Tensor = Q[:, :, :, 2]
    vdo: torch.Tensor = Q[:, :, :, 3]
    tanhe = torch.tanh(Beta * L / 2)

    return torch.sum(
        (udo.conj() * vup - uup.conj() * vdo)
        / 2
        * Potential.view(1, 1, -1)
        * tanhe.unsqueeze(-1),
        dim=1,
    )


@torch.jit.script
def eigenvalue_perturbation_gradient(
    L: torch.Tensor,
    Q: torch.Tensor,
    Beta: torch.Tensor,
    Idx: torch.Tensor,
    Potential: torch.Tensor,
    batch_size: int = 8
):
    """_summary_

    Args:
        L (torch.Tensor): _description_
        Q (torch.Tensor): _description_
        Idx (torch.Tensor): _description_
    """
    K = torch.tensor(
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
    K_Q = torch.matmul(Q_b[:, :, -2 * N :, :], K.view(1, 1, 4, 4))

    Q_b = Q_b.transpose(-2, -1)

    denom = (L.unsqueeze(-1) - L.unsqueeze(-2))
    denom = denom[
        :, -2 * N :, :
    ].unsqueeze(1)  # (B, 1, 2N, 4N). Unsqueeze for the inner batch dimension

    denom = torch.where(denom.abs() < 1e-10, torch.inf, denom)

    # Keep positive eigenvalues
    tanhe = torch.tanh(Beta * L[:, -2 * N :] / 2).view(B, 1, 2*N, 1)

    Q = Q.transpose(-1, -2).view(B, 4 * N, N, 4)
    Q = Q[:, :, Idx, :] # (batch, 4N, nnz, 4)


    uup_0 = Q[:, -2 * N :, :, 0].unsqueeze(1) # (B, 1, 2N, nnz)
    udo_0 = Q[:, -2 * N :, :, 1].unsqueeze(1)
    vup_0 = Q[:, -2 * N :, :, 2].unsqueeze(1)
    vdo_0 = Q[:, -2 * N :, :, 3].unsqueeze(1)

    # diff_E_term = (
    #     Beta
    #     * (udo_0.conj() * vup_0 - uup_0.conj() * vdo_0)
    #     * (1 - tanhe.unsqueeze(-1) ** 2)
    #     / 2
    # )

    Potential = Potential.view(1, 1, 1, -1)#.expand(B, batch_size, 2*N, nnz)
    out = torch.zeros((B, nnz, nnz), dtype=torch.complex128)
    for start in range(0, nnz, batch_size):
        end = int(min(nnz, start + batch_size))
        Q_K_Q = torch.matmul(K_Q[:, start:end, :, :], Q_b[:, start:end, :, :].conj())  # (B, batch_size, 2N, 4N)

        Q_K_Q.div_(denom)


        Q_diff = torch.einsum("baij, bjkl->baikl", Q_K_Q, Q) # (B, batch_size, 2N, nnz, 4)


        # All of these: (B, batch_size, 2N, nnz)
        uup_diff = Q_diff[..., 0]
        udo_diff = Q_diff[..., 1]
        vup_diff = Q_diff[..., 2]
        vdo_diff = Q_diff[..., 3]

        # Maybe include...
        # eval_diff = Q_K_Q.diagonal(dim1=-1, dim2=-2)

        # Perform the sum over the energy dimension
        out[:, start:end, :] = torch.sum(
            (
                (  # First part. Change due to u and v
                    (udo_0.conj() * vup_diff + udo_diff.conj() * vup_0)
                    - (uup_0.conj() * vdo_diff + uup_diff.conj() * vdo_0)
                )
                * tanhe
                # + (  # Second part. Change due to eigenvalue change
                #     diff_E_term * eval_diff.unsqueeze(-1)
                # )
            )
            * Potential
            ,
            dim=2,
        ) / 2

    return out.transpose(-1, -2)
def eigenvalue_perturbation_gradient_v2(
    L: torch.Tensor,
    Q: torch.Tensor,
    Idx: torch.Tensor,
    Beta: torch.Tensor,
    Potential: torch.Tensor,
    batch_size: int = 8
):
    """_summary_

    Args:
        L (torch.Tensor): _description_
        Q (torch.Tensor): _description_
        Idx (torch.Tensor): _description_
    """
    K = torch.tensor(
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
    K_Q = torch.matmul(Q_b[:, :, -2 * N :, :], K.view(1, 1, 4, 4))

    Q_b = Q_b.transpose(-2, -1)

    denom = (L.unsqueeze(-1) - L.unsqueeze(-2))
    denom = denom[
        :, -2 * N :, :
    ].unsqueeze(1)  # (B, 1, 2N, 4N). Unsqueeze for the inner batch dimension

    denom = torch.where(denom.abs() < 1e-10, torch.inf, denom)

    # Keep positive eigenvalues
    tanhe = torch.tanh(Beta * L[:, -2 * N :] / 2).view(B, 1, 2*N, 1)

    Q = Q.transpose(-1, -2).view(B, 4 * N, N, 4)
    Q = Q[:, :, Idx, :] # (batch, 4N, nnz, 4)


    uup_0 = Q[:, -2 * N :, :, 0].unsqueeze(1) # (B, 1, 2N, nnz)
    udo_0 = Q[:, -2 * N :, :, 1].unsqueeze(1)
    vup_0 = Q[:, -2 * N :, :, 2].unsqueeze(1)
    vdo_0 = Q[:, -2 * N :, :, 3].unsqueeze(1)

    # diff_E_term = (
    #     Beta
    #     * (udo_0.conj() * vup_0 - uup_0.conj() * vdo_0)
    #     * (1 - tanhe.unsqueeze(-1) ** 2)
    #     / 2
    # )

    Potential = Potential.view(1, 1, 1, -1)#.expand(B, batch_size, 2*N, nnz)
    out = torch.zeros((B, nnz, nnz), dtype=torch.complex128)
    for start in range(0, nnz, batch_size):
        end = int(min(nnz, start + batch_size))
        Q_K_Q = torch.matmul(K_Q[:, start:end, :, :], Q_b[:, start:end, :, :].conj())  # (B, batch_size, 2N, 4N)

        Q_K_Q.div_(denom)


        Q_diff = torch.einsum("baij, bjkl->baikl", Q_K_Q, Q) # (B, batch_size, 2N, nnz, 4)


        # All of these: (B, batch_size, 2N, nnz)
        uup_diff = Q_diff[..., 0]
        udo_diff = Q_diff[..., 1]
        vup_diff = Q_diff[..., 2]
        vdo_diff = Q_diff[..., 3]

        # Maybe include...
        # eval_diff = Q_K_Q.diagonal(dim1=-1, dim2=-2)

        # Perform the sum over the energy dimension
        out[:, start:end, :] = torch.sum(
            (
                (  # First part. Change due to u and v
                    (udo_0.conj() * vup_diff + udo_diff.conj() * vup_0)
                    - (uup_0.conj() * vdo_diff + uup_diff.conj() * vdo_0)
                )
                * tanhe
                # + (  # Second part. Change due to eigenvalue change
                #     diff_E_term * eval_diff.unsqueeze(-1)
                # )
            )
            * Potential
            ,
            dim=2,
        ) / 2

    return out.transpose(-1, -2)


def profile():
    # Only import if actually use the function...
    from torch.profiler import profile, record_function, ProfilerActivity

    B = 200
    N = 150
    size = 4 * N
    nnz = 80

    L = torch.randn(B, size, dtype=torch.complex128)

    Q = torch.randn(B, size, size, dtype=torch.complex128)
    Idx = torch.arange(nnz)
    Beta = torch.tensor(0.5, dtype=torch.float64)
    Potential = torch.randn(nnz, dtype=torch.complex128)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function("my_eigen_grad_block"):
            out = eigenvalue_perturbation_gradient(L, Q, Idx, Beta, Potential)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


if __name__ == "__main__":
    profile()
