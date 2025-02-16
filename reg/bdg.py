import torch
from dct import dct, idct
import time
from optimization import broydenB2, newton

import matplotlib.pyplot as plt
from gradient import eigenvalue_perturbation_gradient, batch_consistency


class DenseBDGSolver(torch.nn.Module):
    def __init__(
        self,
        H_indices: torch.Tensor,
        H_blocks: torch.Tensor,
        V_indices: torch.Tensor,
        V_potential: torch.Tensor,
        temperature: torch.Tensor,
        kmodes: list[int],
        cosine_threshold: float = 1e-3,
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
        freqs = [2 * torch.cos(2 * torch.pi * torch.fft.fftfreq(N)) for N in kmodes]

        if len(freqs) > 0:
            mesh: list[torch.Tensor] = list(torch.meshgrid(freqs, indexing="ij"))
            kmodes = torch.stack(mesh, dim=0)
            kmodes = kmodes.sum(dim=0).reshape(-1)

            unique_elements, counts = torch.unique(kmodes, return_counts=True)
            self.kmode_weights = (counts / kmodes.shape[0]).to(torch.complex128)
            self.kmodes = unique_elements
        else:
            self.kmode_weights = torch.tensor([1], dtype=torch.complex128)
            self.kmodes = torch.tensor([0], dtype=torch.complex128)

        print(f"Number of unique k modes: {self.kmode_weights.shape}")

        self.diag_mask = torch.diag(torch.tensor([1, 1, -1, -1]).repeat(N))
        self.potential = V_potential

        self.cosine_threshold = cosine_threshold

        self._L = None
        self._Q = None
        self._grad = torch.zeros(
            (self.potential.numel(), self.potential.numel()), dtype=torch.complex128
        )

    def grad(self, x: torch.Tensor, eps: float = 1e-5):
        return self._grad - torch.eye(x.numel(), dtype=torch.complex128)

    def eval(self, x: torch.Tensor):
        return self.zero_func(x)

    # @torch.jit.script_m
    def zero_func(self, x: torch.Tensor):
        # X are cosine wave coefficients. Expand using inverse discrete cosine transform
        # signal = idct(x)
        # res = self.forward(signal)
        # res_x = dct(res)

        # # Keep only cosine modes that fall above some predefined threshold
        # threshold = self.cosine_threshold * res_x.abs().mean().item()
        # res_x[res_x.abs() < threshold] = 0.0
        res_x = self.forward(x)
        return res_x - x

    def insert_deltas(self, x: torch.Tensor):
        # x is of shape (batch, N)
        jsigma2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)
        top_vals = x.view(-1, 1, 1) * jsigma2  # (batch, N, 2, 2)
        bot_vals = top_vals.conj().transpose(-2, -1)

        base_matrix = self.base_matrix
        row = self.potential_rows
        col = self.potential_cols
        base_matrix[row.unsqueeze(-1), col.unsqueeze(-2)] = top_vals
        base_matrix[col.unsqueeze(-1), row.unsqueeze(-2)] = bot_vals
        return base_matrix

    def insert_kmodes(self, base_matrix: torch.Tensor):
        return base_matrix.unsqueeze(0) + self.kmodes.view(
            -1, 1, 1
        ) * self.diag_mask.unsqueeze(0)  # .to(torch.complex64)

    def matrix(self, x: torch.Tensor):
        return self.insert_kmodes(self.insert_deltas(x))

    def consistency_v0(self, L: torch.Tensor, Q: torch.Tensor):
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
        Q = Q[:, :, self.potential_indices, :]  # (B, 2N, nnz, 4)

        # # Store for gradient
        # self._L = torch.clone(L)
        # self._Q = torch.clone(Q)

        # uup: torch.Tensor = Q[:, :, :, 0]  # (num_batches, eigenvalues, position)
        udo: torch.Tensor = Q[:, :, :, 1]
        vup: torch.Tensor = Q[:, :, :, 2]
        # vdo: torch.Tensor = Q[:, :, :, 3] # (num_batches, eigenvalues, position)
        tanhe = torch.tanh(self.beta * L / 2)

        res = self.kmode_weights @ torch.sum(
            (udo.conj() * vup) * tanhe.unsqueeze(-1),
            dim=1,
        )
        return self.potential * res

    def consistency(self, L: torch.Tensor, Q: torch.Tensor):
        return self.kmode_weights @ batch_consistency(
            L, Q, self.potential_indices, self.beta, self.potential
        )

    def calculate_gradient(self, L: torch.Tensor, Q: torch.Tensor, T: torch.Tensor):
        self._grad = (
            self.kmode_weights.view(-1, 1, 1)
            * eigenvalue_perturbation_gradient(
                L,
                Q,
                self.potential_indices,
                Beta= 1.0 / (1e-15 + T),
                Potential=self.potential,
            )
        ).sum(0)  # (B, N, N)
        return self._grad

    def forward(self, x: torch.Tensor):
        # Do batched diagonalisation, one for each k-mode
        print("Diagonalizing...")
        start = time.time()
        L, Q = torch.linalg.eigh(self.matrix(x))
        delta_0 = self.consistency(L, Q)

        mid = time.time()

        print("Starting gradient")

        # Use eigenvalue perturbation to calculate the gradient
        self.calculate_gradient(L, Q, self.temperature)

        end = time.time()

        print(mid - start, end - mid)

        return delta_0

        # TODO: Show equivalency between upper and lower equations
        # return self.kmode_weights @ torch.sum(
        #     (udo.conj() * vup - uup.conj() * vdo)
        #     * self.potential.view(1, 1, -1)
        #     * tanhe.unsqueeze(-1)
        #     / 2,
        #     dim=1,
        # )

    def critical_temperature(self, eps=1e-3):
        L, Q = torch.linalg.eigh(
            self.matrix(torch.zeros_like(self.potential).to(torch.complex128))
        )

        minval = torch.tensor(0.0)
        maxval = torch.tensor(1.0)

        while (maxval - minval) > eps:
            t = (maxval + minval) / 2

            rho = torch.linalg.norm(self.calculate_gradient(L, Q, t), ord=2)

            print(f"{rho.item()}, {minval.item()}, {maxval.item()}")

            if rho < 1:
                maxval = t
            else:
                minval = t





    def free_energy(self, x: torch.Tensor):
        # The gradient obtained using eigvalsh is always numerically stable,
        # as opposed to the ones obtained using eigh.
        evals = torch.linalg.eigvalsh(self.matrix(x))

        # Only the positive eigenvalues contribute to the calculation
        evals = evals[evals > 0]

        # Superconducting contribution
        E0 = torch.dot(x.conj(), x / self.potential)

        # Non-entropic contribution
        H0 = -1 / 2 * torch.sum(evals)

        # Entropy
        S = torch.sum(torch.log(1 + torch.exp(-self.beta * evals)))

        # Final result
        return E0 + H0 - self.temperature * S

    def condensation_energy(self, x: torch.Tensor):
        return self.free_energy(x) - self.free_energy(torch.zeros_like(x))

    def solve(self, return_corr: bool = False):
        before = time.time()

        x0 = torch.ones_like(self.potential).to(torch.complex128)
        # x0[0] = 1.0

        res = newton(self, x0, verbose=True)

        return res.numpy()
        assert False

        # res = broydenB2(self.zero_func, x0, verbose=True, eps=1e-5)
        print(f"Elapsed: {time.time() - before}")
        return res.numpy()
        return idct(res).numpy()
