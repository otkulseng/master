import torch


class DenseBDGSolver(torch.nn.Module):
    def __init__(
        self,
        H_indices: torch.Tensor,
        H_blocks: torch.Tensor,
        V_indices: torch.Tensor,
        V_potential: torch.Tensor,
        temperature: torch.Tensor,
        kmodes: torch.Tensor = torch.tensor(0.0, dtype=torch.complex128),
    ):
        super().__init__()
        # Step 1, generate dense base matrix
        N = int(H_indices[:, 0].max().item() + 1)
        base_matrix = torch.zeros((4 * N, 4 * N), dtype=torch.complex128)
        row = H_indices[:, 0]
        col = H_indices[:, 1]
        blk = H_indices[:, 2]
        top = H_blocks[blk, :, :]  # (N, 2, 2)
        bottom = -top.conj()
        top_row = 4 * row[:, None] + torch.arange(2)  # (N, 2)
        top_col = 4 * col[:, None] + torch.arange(2)  # (N, 2)
        bot_row = top_row + 2
        bot_col = top_col + 2
        base_matrix[top_row.unsqueeze(-1), top_col.unsqueeze(1)] = top
        base_matrix[bot_row.unsqueeze(-1), bot_col.unsqueeze(1)] = bottom
        self.base_matrix = base_matrix

        # Step 2, generate the indexing masks necessary to vectorize insertion of the deltas
        V_row = V_indices[:, 0]
        V_col = V_indices[:, 1]

        # TODO: These only support s-wave
        self.potential_indices = V_row

        self.potential_rows = 4 * V_row[:, None] + torch.arange(2)
        self.potential_cols = 4 * V_col[:, None] + torch.arange(2) + 2

        self.temperature = temperature
        self.beta = 1.0 / (1e-15 + self.temperature)

        # K modes is a up to 3d list of diagonal entries
        freqs = [
            2 * torch.cos(2 * torch.pi * torch.fft.fftfreq(N)) for N in kmodes
        ]

        mesh: list[torch.Tensor] = list(torch.meshgrid(freqs, indexing='ij'))
        kmodes = torch.stack(mesh, dim=0)
        kmodes = kmodes.sum(dim=0).reshape(-1)

        unique_elements, counts = torch.unique(kmodes, return_counts=True)
        self.kmode_weights = (counts / kmodes.shape[0]).to(torch.complex128)
        self.kmodes = unique_elements


        print(f"Number of unique k modes: {self.kmode_weights.shape}")

        self.diag_mask = torch.diag(torch.tensor([1, 1, -1, -1]).repeat(N))

        self.potential = V_potential
    # @torch.jit.script_m
    def zero_func(self, x):
        return self.forward(x) - x

    def forward(self, x: torch.Tensor):
        jsigma2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)
        top_vals = x.view(-1, 1, 1) * jsigma2
        bot_vals = top_vals.conj().transpose(-2, -1)

        base_matrix = self.base_matrix
        row = self.potential_rows
        col = self.potential_cols
        base_matrix[row.unsqueeze(-1), col.unsqueeze(1)] = top_vals
        base_matrix[col.unsqueeze(-1), row.unsqueeze(1)] = bot_vals

        # Do batched diagonalisation, one for each k-mode
        L, Q = torch.linalg.eigh(
            (base_matrix.unsqueeze(0)
            + self.kmodes.view(-1, 1, 1) * self.diag_mask.unsqueeze(0))#.to(torch.complex64)
        )
        # Q = Q.to(torch.complex128)

        # Keep only positive
        mask = L > 0

        # L: (num_batches, 2N) positive eigenvalues
        L = L[mask].view(Q.shape[0], -1)

        # Q: (num_batches, 2N, N, 4) == (num_batches, eigenvalues, position, Nambu)
        Q = Q.transpose(-2, -1)[mask].view(L.shape[0], L.shape[1], -1, 4)
        Q = Q[:, :, self.potential_indices, :] # (n_b, E, pos, N)
        uup = Q[:, :, :, 0]  # (num_batches, eigenvalues, position)
        udo = Q[:, :, :, 1]
        vup = Q[:, :, :, 2]
        vdo = Q[:, :, :, 3]
        tanhe = torch.tanh(self.beta * L / 2)

        return self.kmode_weights @ torch.sum(
            (udo.conj() * vup - uup.conj() * vdo)
            * self.potential.view(1, 1, -1)
            * tanhe.unsqueeze(-1) / 2, dim=1
        )


def main():
    indices = torch.tensor(
        [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=torch.long
    )

    order_params = torch.tensor([1])
    order_params = torch.concat([order_params, -order_params.conj()])

    indices = torch.tensor(
        [
            [1, 1],
            # [1, 1]
        ]
    )

    rowcol = bdg_order_matrix_indices(indices)

    shape = (1, 4, 2)
    vec = torch.randn(size=shape).to(torch.float32)
    res = apply_order_parameters(rowcol, order_params, vec)

    print(res)
    assert False

    blocks = torch.tensor(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [[5, 6], [7, 8]],
        ]
    )
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
