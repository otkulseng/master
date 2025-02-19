import torch
from dct import dct, idct
import time
from optimization import broydenB2, newton

import matplotlib.pyplot as plt
from gradient import eigenvalue_perturbation_gradient, batch_consistency

from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

import storage
class DenseBDGSolver(torch.nn.Module):
    def __init__(
        self,
        H_indices: torch.Tensor,
        H_blocks: torch.Tensor,
        V_indices: torch.Tensor,
        V_potential: torch.Tensor,
        kmodes: list[int],
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
                Beta=1.0 / (1e-15 + T),
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

    def find_blocks(self, matrix: torch.Tensor):
        B, N, N = matrix.shape

        mask = torch.where(matrix[0].abs() > 0, 1, 0).numpy()
        csgraph = sp.coo_array(mask)

        num_blocks, labels = connected_components(csgraph=csgraph, directed=False)
        if num_blocks == 1:
            # Default to regular in the case of no blocks
            return matrix.unsqueeze(1)

        indices = torch.arange(N)
        blocks = [indices[labels == i] for i in range(num_blocks)]

        new_matrix = torch.stack(
            [matrix[:, blk.unsqueeze(-1), blk.unsqueeze(-2)] for blk in blocks], dim=1
        ).to(matrix.dtype)
        return new_matrix, blocks

    def block_diagonalize(self, matrix: torch.Tensor):
        B, N, N = matrix.shape
        print("Finding blocks")
        matrix, blocks = self.find_blocks(matrix)

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

    def critical_temperature(self, minval=0.0, maxval=1.0, eps=1e-3):
        print("Diagonalizing")
        L, Q = self.block_diagonalize(
            self.matrix(torch.zeros_like(self.potential.to(torch.complex128)))
        )

        print("Starting binary search")
        minval = torch.tensor(minval)
        maxval = torch.tensor(maxval)

        # Indicator function. Largest eigenvalue less than 1 means non-superconducting.
        # That is, need to make maxval smaller

        def indicator_func(t):
            rho = torch.linalg.norm(self.calculate_gradient(L, Q, t), ord=2)
            print(rho)
            return rho < 1

        # Start in the middle
        r = 0.5
        # Go aggressively down from maxval
        rmax = 1
        rmin = 0

        print("Entering first loop")
        while indicator_func(minval + (maxval - minval) * r):
            # This value of r is too big. New maximum
            rmax = r
            r = r**2

            print(r)

            if r < 1e-10:
                return minval

        # Have now found a value of r that is too small.
        rmin = r


        new_min = minval + (maxval - minval)*rmin
        new_max = minval + (maxval - minval)*rmax

        minval = new_min
        maxval = new_max

        print("Entering second loop")
        while (maxval - minval) / (maxval + minval) > eps:
            t = (maxval + minval) / 2

            rho = indicator_func(t)

            print(f"{minval.item()}, {maxval.item()}")

            if rho < 1:
                maxval = t
            else:
                minval = t

            storage.save_kwargs('crit_temps', minval=minval.item(), maxval=maxval.item())
        return (maxval + minval).item() / 2

    def free_energy(self, x: torch.Tensor):
        # The gradient obtained using eigvalsh is always numerically stable,
        # as opposed to the ones obtained using eigh.
        evals = torch.linalg.eigvalsh(self.find_blocks(self.matrix(x)))

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

    def solve(self, temperature: float):
        before = time.time()

        self.temperature = torch.tensor(temperature)
        self.beta = 1.0 / (1e-15 + self.temperature)

        x0 = torch.ones_like(self.potential).to(torch.complex128)
        # x0[0] = 1.0

        res = newton(self, x0, verbose=True)
        storage.store('order_parameters', res)

        return res.numpy()
        assert False

        # res = broydenB2(self.zero_func, x0, verbose=True, eps=1e-5)
        print(f"Elapsed: {time.time() - before}")
        return res.numpy()
        return idct(res).numpy()
