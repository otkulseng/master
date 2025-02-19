import torch

from scipy.optimize import approx_fprime
import numpy as np
import storage

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
    f,
    x0,
    max_iter=100,
    eps=1e-10,
    verbose=False,
    rel_eps=1e-10,
    x_norm=1e-10,
):

    best_norm = torch.inf

    for it in range(max_iter):
        f0 = f.eval(x0)

        current_norm = torch.max(torch.abs(f0))
        rel_change = torch.max(torch.abs(f0) / (1e-15 + torch.abs(x0)))

        if current_norm < best_norm:
            best_norm = current_norm
            best_x = torch.clone(x0)

        if verbose:
            print(
                f"It: {it}\t norm: {current_norm}\t rel: {rel_change}\t mid: {torch.mean(torch.abs(x0))}"
            )

        if current_norm < eps:
            break

        if rel_change < rel_eps:
            break

        if torch.max(torch.abs(best_x)) < x_norm:
            break

        # gr = torch.clone(f.grad(x0)).real
        # def inner(x):
        #     x = torch.tensor(np.real(x)).to(torch.complex128)
        #     return torch.real(f.eval(x)).numpy()
        # jac = approx_fprime(x0, inner)
        # # Perform newton step

        # print(gr)
        # print(jac)
        # print((gr - jac).abs())
        # # print(jac)
        # print("Largest: ", (gr - torch.tensor(jac)).abs().max().item())
        # assert(False)



        x0 = x0 + torch.linalg.solve(f.grad(x0), -f0)

        storage.store('newton_x', x0)
        storage.store('newton_f', f0)
    return best_x


