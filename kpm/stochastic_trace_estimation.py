import torch


class LinearOperator:
    def __init__(self, tensor: torch.Tensor):
        self.matr = tensor
    def matvec(self, other):
        return other @ self.matr.T

    def __call__(self, x, *args, **kwds):
        return self.matvec(x)

def chebyshev_stochastic_estimation(
        operator: LinearOperator,
        chebyshev_coefficients: torch.Tensor,
        vectors: torch.Tensor
    ):
    # Ordinary chebyshev

    phi0 = torch.clone(vectors)
    phi2 = 2 * operator(operator(phi0)) - phi0

    print(vectors.shape, phi0.shape)
    total = torch.einsum("bi, bi->b", vectors, phi0) * chebyshev_coefficients[0]
    total += torch.einsum("bi, bi->b", vectors, phi2) * chebyshev_coefficients[1]
    for N in range(2, chebyshev_coefficients.size):
        phi4 = 2 * (2 * operator(operator(phi2)) - phi2) - phi0

        total += torch.einsum("bi, bi->b", vectors, phi4) * chebyshev_coefficients[N]
        phi0 = phi2
        phi2 = phi4

    total = torch.mean(total)
    return total


def clenshaw_chebyshev_stochastic_estimation(
        operator: LinearOperator,
        chebyshev_coefficients: torch.Tensor,
        vectors: torch.Tensor
    ):

    phi2 = torch.zeros_like(vectors)
    phi1 = torch.zeros_like(vectors)

    for N in range(chebyshev_coefficients.size - 1, 0, -1):
        phi0 = chebyshev_coefficients[N]*vectors + 2 * (2 * operator(operator(phi1)) - phi1) - phi2
        phi2 = torch.clone(phi1)
        phi1 = torch.clone(phi0)

    result = vectors * chebyshev_coefficients[0] - phi2 + (2 * operator(operator(phi1)) - phi1)

    return torch.mean(
        torch.einsum("bij, bij->b", vectors, result)
    )






