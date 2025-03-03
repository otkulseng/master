import torch
from dct import dct, idct
import time
from optimization import broydenB2, newton

import matplotlib.pyplot as plt
from gradient import eigenvalue_perturbation_gradient, batch_consistency


import storage
from util import block_diagonalize, BDGFunction


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
        self.base_matrix: torch.Tensor = base_matrix  # (1, 4N, 4N)

        # Step 2, generate the indexing masks necessary to vectorize insertion of the deltas
        V_row = V_indices[:, 0]
        V_col = V_indices[:, 1]

        # TODO: These only support s-wave
        self.potential_indices = V_row

        self.potential_rows = 4 * V_row[:, None] + torch.arange(2)
        self.potential_cols = 4 * V_col[:, None] + torch.arange(2) + 2

        self.potential_rows = self.potential_rows  # Batch dimension
        self.potential_cols = self.potential_cols  # Batch dimension

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

        self.diag_mask = torch.diag(torch.tensor([1, 1, -1, -1]).repeat(N)).unsqueeze(
            0
        )  # (1, N, N)
        self.potential = V_potential

        self._L = None
        self._Q = None
        self._grad = torch.zeros(
            (self.potential.numel(), self.potential.numel()), dtype=torch.complex128
        )

    def solve_diagonals(self, diags: torch.Tensor, temp: torch.Tensor):
        """Adds each number in diags to the diagonal in a diag-mask sort of way

        Args:
            diags (torch.Tensor): _description_
        """

        # Step 1. Create all the matrices stemming from broadcasting each element of diags on the diagonal
        # of self.base_matrix
        beta = 1.0 / (1e-15 + temp)
        base_matrix = self.base_matrix.clone()
        size, _ = base_matrix.shape
        N = size // 4
        B = diags.shape[0]
        diag_mask = torch.diag(torch.tensor([1, 1, -1, -1]).repeat(N)).unsqueeze(
            0
        )  # (1, 4N, 4N)
        all_matrices = diags.view(-1, 1, 1) * diag_mask + base_matrix.unsqueeze(
            0
        )  # (B, 4N, 4N)

        # Step 2. Calculate gradient around Delta=0, to check which diags will give Delta_k = 0
        L, Q = block_diagonalize(all_matrices)
        grads = eigenvalue_perturbation_gradient(
            L, Q, Beta=beta, Idx=self.potential_indices, Potential=self.potential
        )
        # Calculate the largest singular value for each batch dimension.
        # When this is above 1, it means that Delta=0 is stable at this temperature
        # and therefore we know immediately that this is the solution.
        # TODO: Might include trig-expansion here. Intuitively, that would strengthen the above argument.
        # TODO: Check whether matrix_norm could be used. They use A.abs() which would ruin...
        rho = torch.linalg.norm(grads, ord=2, dim=(-1, -2))
        # print(rho)
        mask = rho > 1  # These are the ones that will not converge to 0
        # Step 3, create a BDGFunction whose zeros are the deltas
        # func = torch.jit.script(BDGFunction(
        #     all_matrices[mask], # Keep only nonzero delta_k's
        #     beta,
        #     self.potential_indices,
        #     self.potential,
        # ))

        # Step 4, use newtons method to solve for the zero

        # x0 = newton(
        #     func,
        #     torch.ones(self.potential.numel(), dtype=torch.complex128).unsqueeze(0),
        #     verbose=True
        # )

        func = torch.jit.script(BDGFunction(
            all_matrices[mask], # Keep only nonzero delta_k's
            beta,
            self.potential_indices,
            self.potential,
        ))
        # for a, b in zip(x0, rho[mask]):
        #     print(a.abs().sum().item(), b)

        # Step 4, use newtons method to solve for the zero

        x = newton(
            func,
            torch.ones(self.potential.numel(), dtype=torch.complex128).unsqueeze(0),
            verbose=True
        )

        out = torch.zeros((B, x.size(-1)), dtype=torch.complex128)
        out[mask] = x

        return out

    def solve_integral(self, temp: torch.Tensor):
        # from integration import kronrod61_w, kronrod61_x, gauss_30_w
        # a, b = 0, torch.pi
        # # Transform integral from -1 to 1 to 0 to pi

        # kronrod_nodes = (kronrod61_x.to(torch.complex128) + 1) / 2 * (b-a) + a
        # kronrod_weights = kronrod61_w.to(torch.complex128) * (b - a) / 2
        # gauss_30_w = gauss_30_w.to(torch.complex128)* (b - a) / 2

        # kmodes = 2 * torch.cos(kronrod_nodes)

        # res = self.solve_diagonals(kmodes, temp)

        # full_precision = kronrod_weights @ res / torch.pi
        # half_precision = gauss_30_w.to(torch.complex128) @ res[1::2] / torch.pi



        # Combined chebyshev-gauss, trapezoidal rule

        # Subdivide 0 to pi in N equal pieces

        N = 201
        kvals = torch.pi * (torch.arange(N) * 2 + 1) / (2 * N)
        weights = torch.ones(N, dtype=torch.complex128) / N
        # weights[0] = 0.5
        # weights[-1] = 0.5

        # Take the chebyshev-nodes
        nodes = 2 * torch.cos(kvals)

        res = self.solve_diagonals(nodes, temp) # (B, nnz)

        diff = torch.diff(res, dim=0).norm(dim=1, p=torch.inf) / res.abs().max() # diff along batch dimension
        full_precision = weights @ res
        # print(diff.shape)
        # assert(False)


        plt.figure()
        plt.plot((nodes[1:] + nodes[:-1])/4, diff.numpy())
        plt.savefig("precision.pdf")
        plt.figure()

        return full_precision

    def grad(self, x: torch.Tensor, eps: float = 1e-5):
        return self._grad - torch.eye(self._grad.shape[-1], dtype=torch.complex128)

    def eval(self, x: torch.Tensor, mask: torch.Tensor):
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

    def insert_deltas_old(self, x: torch.Tensor):
        # x is of shape (batch, N)
        B, nnz = x.shape
        jsigma2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)
        # jsigma2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)
        # top_vals = x.view(-1, 1, 1) * jsigma2  # (batch, N, 2, 2)
        top_vals = x.view(B, -1, 1, 1) * jsigma2  # (batch, N, 2, 2)

        bot_vals = top_vals.conj().transpose(-2, -1)

        # base_matrix = self.base_matrix
        row = self.potential_rows  # (B, nnz,)
        col = self.potential_cols

        # print(row.shape, col.shape, base_matrix.shape )
        base_matrix = self.base_matrix.expand(B, -1, -1).clone()  # (B, 4N, 4N)
        base_matrix[:, row.unsqueeze(-1), col.unsqueeze(-2)] = top_vals
        base_matrix[:, col.unsqueeze(-1), row.unsqueeze(-2)] = bot_vals

        # Returns base_matrix of shape (B, N, N)
        return base_matrix

    def insert_kmodes(self, base_matrix: torch.Tensor):
        return (
            base_matrix + self.kmodes.view(-1, 1, 1) * self.diag_mask
        )  # .to(torch.complex64)

    def matrix(self, x: torch.Tensor):
        return self.insert_kmodes(self.insert_deltas(x))

    def consistency(self, L: torch.Tensor, Q: torch.Tensor):
        # No batch average yet
        return batch_consistency(
            L, Q, self.potential_indices, self.beta, self.potential
        )

    def calculate_gradient(self, L: torch.Tensor, Q: torch.Tensor, T: torch.Tensor):
        # self._grad = ( Old
        #     self.kmode_weights.view(-1, 1, 1)
        #     * eigenvalue_perturbation_gradient(
        #         L,
        #         Q,
        #         self.potential_indices,
        #         Beta=1.0 / (1e-15 + T),
        #         Potential=self.potential,
        #     )
        # ).sum(0)  # (N, N)

        # Batched gradient
        self._grad = eigenvalue_perturbation_gradient(
            L,
            Q,
            self.potential_indices,
            Beta=1.0 / (1e-15 + T),
            Potential=self.potential,
        )
        # (B, N, N)
        return self._grad

    def forward(self, x: torch.Tensor):
        # Do batched diagonalisation, one for each k-mode
        # x is either of shape (B, nnz) or (1, nnz). Either case broadcastable to (1, nnz)
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

        if indicator_func(minval):
            # Non-superconducting at minval...
            return minval

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
            rmin = r

            print(r)

            if r < 1e-10:
                return minval.item()

        # # Have now found a value of r that is too small.
        # rmin = r

        new_min = minval + (maxval - minval) * rmin
        new_max = minval + (maxval - minval) * rmax

        minval = new_min
        maxval = new_max

        print("Entering second loop")
        while (maxval - minval) / (maxval + minval) > eps:
            t = (maxval + minval) / 2

            print(f"{minval.item()}, {maxval.item()}")

            if indicator_func(t):
                maxval = t
            else:
                minval = t

            storage.save_kwargs(
                "crit_temps", minval=minval.item(), maxval=maxval.item()
            )
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

        # Shape (1, N) to be broadcastable to batch
        x0 = torch.ones_like(self.potential).to(torch.complex128).unsqueeze(0)
        # x0[0] = 1.0

        res = newton(self, x0, verbose=True)
        storage.store("order_parameters", res)

        print(res.shape)

        res = self.kmode_weights @ res

        return res.numpy()
        assert False

        # res = broydenB2(self.zero_func, x0, verbose=True, eps=1e-5)
        print(f"Elapsed: {time.time() - before}")
        return res.numpy()
        return idct(res).numpy()
