import torch

from scipy.optimize import approx_fprime
import numpy as np
import storage
from util import BDGFunction

def broydenB2(
    f: BDGFunction,
    x0,
    x1=None,
    J0=None,
    max_iter=100,
    eps=1e-10,
    verbose=False,
    rel_eps=1e-4,
    x_norm=1e-10,
):
    f0: torch.Tensor = f.precond_call(x0)

    if J0 is None:
        J0 = torch.eye(f0.shape[-1], dtype=f0.dtype).unsqueeze(0).expand(f0.shape[0], -1, -1)


    if x1 is None:
        x1 = x0 - torch.linalg.solve(J0, f0)

    best_norm = torch.inf

    for it in range(max_iter):
        f1 = f.precond_call(x1)

        current_norm: torch.Tensor = torch.linalg.vector_norm(f1, dim=-1, ord=torch.inf)
        current_norm = current_norm.mean()

        if current_norm < best_norm:
            best_norm = current_norm
            best_x = torch.clone(x1)

        if verbose:
            print(
                f"It: {it}\t norm: {current_norm}\t \t mid: {torch.mean(torch.abs(x1))}"
            )

        if current_norm < eps:
            break

        df = f1 - f0 # (B, N)
        dx = x1 - x0 # (B, N)

        numerator = (dx - (J0 @ df.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1) * df.unsqueeze(-2)
        denom: torch.Tensor = (df * df).sum(-1)



        J0 = J0 + numerator / denom.view(-1, 1, 1)

        x2 = x1 - (J0 @ f1.unsqueeze(-1)).squeeze(-1)

        x0 = x1
        x1 = x2
        f0 = f1

    return best_x


def broydenB1(
    f: BDGFunction,
    x0,
    x1=None,
    J0=None,
    max_iter=100,
    eps=1e-10,
    verbose=False,
    rel_eps=1e-4,
    x_norm=1e-10,
):
    f0 = f(x0)

    if J0 is None:
        J0 = torch.eye(f0.numel(), dtype=f0.dtype)

    if x1 is None:
        x1 = x0 - torch.linalg.solve(J0, f0)

    best_norm = torch.inf

    for it in range(max_iter):
        f1 = f(x1)

        current_norm = torch.max(torch.abs(f1))
        rel_change = torch.max(torch.abs(f1) / (1e-15 + torch.abs(x1)))

        if current_norm < best_norm:
            best_norm = current_norm
            best_x = torch.clone(x1)

        if verbose:
            print(
                f"It: {it}\t norm: {current_norm}\t rel: {rel_change}\t mid: {torch.mean(torch.abs(x1))}"
            )

        if current_norm < eps:
            break

        if rel_change < rel_eps:
            break

        if torch.max(torch.abs(best_x)) < x_norm:
            break

        df = f1 - f0
        dx = x1 - x0
        J0 = J0 + torch.outer((df - J0 @ dx), dx) / (torch.dot(dx, dx))

        x2 = x1 + torch.linalg.solve(J0, -f1)

        x0 = x1
        x1 = x2
        f0 = f1

    return best_x


# 8 matriser av str 11x12 = M ([100], 11, 12)
# 8 vektorer av str 12 = V ([100], 12)
# M V = ([100], 11)


# M = (100, 11, 12)
# V = (100, 12)

# A = torch.zeros_like(M)  (100, 11, 12) Samme shape, samme dtype men alle elementer lik 0
# B = torch.empty_like(M)
# torch.bmm(M, V) -> (100, 11)
# torch/jax generaliseringer av numpy.



def newton(
    f: BDGFunction,
    x0,
    max_iter=100,
    eps=1e-10,
    verbose=False,
    rel_eps=1e-10,
    x_norm=1e-10,
):

    # This is a batched newtons method!
    # x0 is of shape (B, N)


    # Do empty_like f0, as x0 might be (1, nnz) due to broadcasting capabilities.
    # f0 is always (B, nnz).
    # Storage for output
    f0, J0 = f(x0)

    x = x0 - torch.linalg.solve(J0, f0)
    # Therefore, do one iteration.

    # f0 (1000, 200, 100, 20, 300)
    # A sin dim = (100, 200)
    # A[1, 1] = 10
    # A[1, 2] = 30
    # B sin dim = (10, 20, 30)
    # B[4, 2, 5] = 4
    #
    out = torch.zeros_like(f0)

    B, _ = f0.shape
    active_indices = torch.arange(B)
    for it in range(max_iter):
        # Continue until all converge
        if len(active_indices) == 0:
            break

        x_active: torch.Tensor = x[active_indices]

        f_active, J_active = f(x_active, active_indices)
        # norm_active: torch.Tensor = torch.linalg.norm(f_active,dim=-1)
        norm_active: torch.Tensor = torch.linalg.vector_norm(f_active, dim=-1, ord=torch.inf)
        converged = norm_active < eps

        x_active_new = x_active +  torch.linalg.solve(J_active, -f_active)
        x[active_indices] = x_active_new

        # Store the ones that converged
        out[active_indices[converged]] = x_active_new[converged]

        active_indices = active_indices[~converged]

        # print(active_indices)


        if verbose:
            print(
                f"It: {it}\t norm: {norm_active.max().item()}\t #active: {active_indices.numel()}\t mid: {x_active.abs().mean().item()}"
            )
        storage.store('newton_x', out)
        storage.store('newton_f', f_active)
    return out


def preconditioned_broydenB2(
    f: BDGFunction,
    x0,
    max_iter=100,
    eps=1e-10,
    verbose=False,
    rel_eps=1e-10,
    x_norm=1e-10,
):

    # Precondition f with inverse of J

    def new_f(x):
        f0, J0 = f(x)
        return torch.linalg.solve(J0, f0)

    return broydenB2(
        new_f,
        x0,
        max_iter=max_iter,
        eps=eps,
        verbose=verbose,
        rel_eps=rel_eps
    )