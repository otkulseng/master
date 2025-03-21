import jax
import jax.numpy as jnp
from typing import NamedTuple


class OptimResult(NamedTuple):
    x: jax.Array
    fx: jax.Array
    iterations: int
    Jinv: jax.Array


def stable_newton(func, x0, eps=1e-10, max_iter=100):
    # First, ordinary newton and remove all that fail the armijo rule

    # alpha = 1
    # Calc percentage done illegal


    def cond_func(state):
        i, x, _, done = state
        return (i < max_iter) & ~jnp.all(done)

    def body_fun(state):
        i, x, fx_old, done = state
        mask = ~done

        if i % 10 == 0:
            # Start with initial condition smaller
            val = 10 ** (-(i // 10))
            print(f"{i}: Setting: {jnp.sum(mask)} values to {val}")
            x = x.at[mask].set(val)

        fx, jx = func(x[mask], mask)

        norm = jnp.linalg.norm(fx, axis=-1)
        new_done = norm < eps

        dx = -jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)
        # xn = x

        # Now, set all that fail the armijo rule as 'done'
        fx_norm = jnp.linalg.norm(fx_old[mask], axis=-1)

        # norm_old = jnp.linalg.norm(fx_old, axis=-1)
        illegal_norm = fx_norm / norm
        illegal = illegal_norm < 0 # Remove this possibility

        # print(jnp.sum(illegal))

        new_done = jnp.logical_or(illegal, new_done)
        done = done.at[mask].set(new_done)
        print(i, jnp.mean(norm), jnp.mean(done), jnp.sum(illegal))

        # print(done.shape, illegal.shape, fx_norm.shape)
        # print(fx_old.shape)

        return i + 1, x.at[mask].add(dx), fx_old.at[mask].set(fx), done

    init_val = (0, x0, x0, jnp.zeros(x0.shape[0], dtype=bool))

    val = init_val
    while cond_func(val):
        val = body_fun(val)

    it, x, fx, _ = val

    # solved:jax.Array = jnp.array(jnp.linalg.norm(fx, axis=-1) < eps)

    # if ~jnp.all(solved):
    #     # Solve with safer newton method

    #     unsolved = ~solved
    #     x0 = x[unsolved]

    #     # res_broyden = broyden_b1(func, x, eps=eps, max_iter=100, done=~)
    #     res_broyden = broyden_b1(func, x0, eps=eps, max_iter=100, solved=solved)

    #     # x = res_broyden.x
    #     # fx = res_broyden.fx
    #     x = x.at[unsolved].set(res_broyden.x)
    #     fx = fx.at[unsolved].set(res_broyden.fx)

    print(f"Number of unsolved: {jnp.sum(jnp.linalg.norm(fx, axis=-1) > eps)}")
    return OptimResult(x, fx, it, None)

def safeguarded_newton(func, x0, eps=1e-10, max_iter=100, done=None):
    if done is None:
        done = jnp.zeros(x0.shape[0], dtype=bool)

    print(f"Solving: {x0.shape[0]} using safeguarded newton")
    x = x0
    fx, jx = func(x, ~done)
    tikhonov_lambda = 1e-4 * jnp.mean(jnp.abs(x0))

    for it in range(max_iter):
        if jnp.all(done):
            break

        # Find search direction. Add lambda for regularization.
        # Find
        px = - jnp.linalg.solve(jx + tikhonov_lambda, fx[..., None]).squeeze(-1)

        # Find
        fx, jx = func(x + px, ~done)

    print(f"Took: {it} iterations.")
    return OptimResult(x, fx, it, None)

def broydenb2(both_func, x0, eps=1e-10, max_iter=20, mask=None):
    if mask is None:
        mask = jnp.ones(x0.shape[0], dtype=bool)
    def temp(x):
        fx, jx = both_func(x, mask)
        return jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)

    func = jax.jit(temp)

    def cond_fun(state):
        i, x, fx, H, done = state
        return (i < max_iter) & ~jnp.all(done)

    def body_fun(state):
        i, x, fx, H, done = state

        xn = x - jnp.matmul(H, fx[..., None]).squeeze(-1)
        # xn: jax.Array = x
        fxn = func(xn)

        dx: jax.Array = xn - x
        df: jax.Array = fxn - fx

        Hdf = jnp.matmul(H, df[..., None]).squeeze(-1)  # (B, N)
        numer = jnp.expand_dims((dx - Hdf), -1) * jnp.expand_dims(df, -2)  # (B, N)
        denom = jnp.sum(df.conj() * df, axis=-1)  # (B, )

        dHn = numer / denom[:, None, None]  # (B, N, N)

        norm = jnp.linalg.norm(fx, axis=-1)
        print(jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn)  # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps

        return i + 1, xnext, fxn, Hnext, norm < eps

    f0 = func(x0)
    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N)),
        jnp.zeros(B, dtype=bool),
    )
    print(f"Solving: {B} matrices using broyden's method.")
    state = init_val
    while cond_fun(state):
        state = body_fun(state)
    # state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux

    print(f"Took: {i} iterations")
    return OptimResult(x, fx, i, H)


def broyden_b1(both_func, x0: jax.Array, eps=1e-10, max_iter=100, solved=None):

    inner_mask = jnp.zeros_like(solved)
    def func(x, mask):
        new_mask = inner_mask.at[~solved].set(mask)
        fx, jx = both_func(x, new_mask)
        return jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)

    done = jnp.zeros(x0.shape[0], dtype=bool)
    # func = jax.jit(inner_func)

    # Now, callable f
    f0: jax.Array = func(x0, ~done)
    B, N = f0.shape

    print(f"Solving {B} matrices using broyden's diminishing first method")
    jac = jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N))

    alpha = 1.0

    for it in range(max_iter):
        if jnp.all(done):
            break
        mask = ~done



        # Only call function on the ones that are not done
        # Commensurate shapes
        jac_cur = jac[mask]
        x0_cur = x0[mask]
        f0_cur = f0[mask]

        x = x0_cur - jnp.linalg.solve(jac_cur, f0_cur[..., None]).squeeze(-1)
        fx = func(x, mask)

        dx = x - x0_cur
        df = fx - f0_cur

        Hdx = jnp.matmul(jac_cur, dx[..., None]).squeeze(-1)  # (nnz, N)
        numer = jnp.expand_dims((df - Hdx), -1) * jnp.expand_dims(dx, -2)  # (mmz, N)
        denom = jnp.sum(dx.conj() * dx, axis=-1)  # (nnz, )
        dHn = numer / denom[:, None, None]  # (nnz, N, N)

        norm = jnp.linalg.norm(dx, axis=-1)
        new_done = norm < eps

        jac = jac.at[mask].add(dHn)
        done = done.at[mask].set(new_done)
        print(it, jnp.mean(norm), jnp.mean(done))

        x0 = x0.at[mask].set(x)
        f0 = f0.at[mask].set(fx)
    return OptimResult(x0, f0, it, None)







def fdshasksjldh(both_func, x0, eps=1e-10, max_iter=20, mask=None):
    if mask is None:
        mask = jnp.zeros(x0.shape[0], dtype=bool)

    def temp(x):
        fx, jx = both_func(x[mask], mask)
        return jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)

    func = jax.jit(temp)

    def cond_fun(state):
        i, x, fx, H, done = state
        return (i < max_iter) & ~jnp.all(done)


    def body_fun(state):
        i, x, fx, H, done = state

        xn = x - jnp.linalg.solve(H, fx[..., None]).squeeze(-1)
        # xn: jax.Array = x
        fxn = func(xn)

        dx: jax.Array = xn - x
        df: jax.Array = fxn - fx

        Hdx = jnp.matmul(H, dx[..., None]).squeeze(-1)  # (B, N)
        numer = jnp.expand_dims((df - Hdx), -1) * jnp.expand_dims(dx, -2)  # (B, N)
        denom = jnp.sum(dx.conj() * dx, axis=-1)  # (B, )

        dHn = numer / denom[:, None, None]  # (B, N, N)

        norm = jnp.linalg.norm(dx, axis=-1)
        print(i, jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn)  # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps

        return i + 1, xnext, fxn, Hnext, norm < eps

    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        f0,
        ~mask,
    )
    print(f"Solving: {B} matrices using broyden's method.")
    state = init_val
    while cond_fun(state):
        state = body_fun(state)
    # state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux

    print(f"Took: {i} iterations")
    return OptimResult(x, fx, i, H)


def dynamic_broyden(func, x0, eps=1e-10, outer_iter=10, inner_iter=10):
    def cond_fun(state):
        i, x, fx, H, done = state
        return (i < max_iter) & ~jnp.all(done)

    def body_fun(state):
        i, x, fx, H, done = state

        xn = x - jnp.matmul(H, fx[..., None]).squeeze(-1)
        # xn: jax.Array = x
        fxn = func(xn)

        dx: jax.Array = xn - x
        df: jax.Array = fxn - fx

        Hdf = jnp.matmul(H, df[..., None]).squeeze(-1)  # (B, N)
        numer = jnp.expand_dims((dx - Hdf), -1) * jnp.expand_dims(df, -2)  # (B, N)
        denom = jnp.sum(df.conj() * df, axis=-1)  # (B, )

        dHn = numer / denom[:, None, None]  # (B, N, N)

        norm = jnp.linalg.norm(fx, axis=-1)
        # print(jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn)  # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps

        return i + 1, xnext, fxn, Hnext, norm < eps

    f0 = func(x0)
    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N)),
        jnp.zeros(B, dtype=bool),
    )
    # state = init_val
    # while cond_fun(state):
    #     state = body_fun(state)
    state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux
    return OptimResult(x, fx, i, H)


def broydenb2symmetric(func, x0, eps=1e-10, max_iter=20):
    # Symmetric
    def cond_fun(state):
        i, x, fx, H, done = state
        return (i < max_iter) & ~jnp.all(done)

    def body_fun(state):
        i, x, fx, H, done = state

        xn = x - jnp.matmul(H, fx[..., None]).squeeze(-1)
        # xn: jax.Array = x
        fxn = func(xn)

        dx: jax.Array = xn - x
        df: jax.Array = fxn - fx

        first_term = (
            jnp.expand_dims(dx, axis=-1)
            * jnp.expand_dims(dx, axis=-2)
            / jnp.sum(dx.conj() * df, axis=-1)[:, None, None]
        )

        n_v = jnp.matmul(H, df[..., None]).squeeze(-1)

        second_numerator = jnp.expand_dims(n_v, axis=-1) * jnp.expand_dims(n_v, axis=-2)
        second_denom = jnp.sum(df.conj() * n_v, axis=-1)
        second_term = -second_numerator / second_denom[:, None, None]

        dHn = first_term + second_term

        norm = jnp.linalg.norm(fx, axis=-1)
        print(jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn)  # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps

        return i + 1, xnext, fxn, Hnext, norm < eps

    f0 = func(x0)
    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N)),
        jnp.zeros(B, dtype=bool),
    )
    state = init_val
    while cond_fun(state):
        state = body_fun(state)
    # state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux
    return (x, OptimResult(i, fx, H))
    return state
