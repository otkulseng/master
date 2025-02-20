import torch

from scipy.optimize import approx_fprime
import numpy as np
import storage
from util import BDGFunction

def broydenB2(
    f,
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

        current_norm = torch.linalg.norm(f1)
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

        J0 = J0 + torch.outer((dx - J0 @ df), df) / (torch.dot(df, df))

        x2 = x1 - J0 @ f1

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

    out = torch.empty_like(f0)

    B, _ = f0.shape
    active_indices = torch.arange(B)
    for it in range(max_iter):
        # Continue until all converge
        if len(active_indices) == 0:
            break

        x_active = x[active_indices]

        f_active, J_active = f(x_active, active_indices)
        norm_active: torch.Tensor = torch.linalg.vector_norm(f_active, dim=-1, ord=torch.inf)
        converged = norm_active < eps

        x_active_new = x_active + torch.linalg.solve(J_active, -f_active)
        x[active_indices] = x_active_new

        # Store the ones that converged
        out[active_indices[converged]] = x_active_new[converged]

        active_indices = active_indices[~converged]

        # print(active_indices)


        if verbose:
            print(
                f"It: {it}\t norm: {norm_active.max().item()}\t #active: {active_indices.numel()}\t mid: {torch.mean(torch.abs(x0))}"
            )
        storage.store('newton_x', out)
        storage.store('newton_f', f_active)
    return out


