import jax
import jax.numpy as jnp
from typing import NamedTuple

class OptimResult(NamedTuple):
    x: jax.Array
    fx: jax.Array
    iterations: int
    Jinv: jax.Array

def stable_newton(func, x0, eps=1e-10, max_iter=100):
    # Ensure function is jitted
    # jit_func = jax.jit(func)

    # First, do 5

    # for it in range(max_iter):
    #     x, fx = None, None

    def cond_func(state):
        i, x, _, done = state
        return (i < max_iter) & ~jnp.all(done)

    def body_fun(state):
        i, x, _, done = state
        mask = ~done
        fx, jx = func(x, mask)

        norm = jnp.linalg.norm(fx, axis=-1)
        new_done = norm < eps
        print(i, jnp.mean(norm), jnp.mean(done))

        dx = - 0.5 * jnp.linalg.solve(jx, fx[..., None]).squeeze(-1)
        # xn = x

        done = done.at[mask].set(new_done)
        return i+1, x.at[mask].add(dx), fx, done

    init_val = (
        0,
        x0,
        x0,
        jnp.zeros(x0.shape[0], dtype=bool)
    )

    val = init_val
    while cond_func(val):
        val = body_fun(val)


    it, x, fx, _ = val
    return OptimResult(x, fx, it, None)





def broydenb2(both_func, x0, eps=1e-10, max_iter=20):

    def temp(x):
        fx, jx = both_func(x)
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


        Hdf = jnp.matmul(H, df[..., None]).squeeze(-1) # (B, N)
        numer = jnp.expand_dims((dx - Hdf), -1) * jnp.expand_dims(df, -2) # (B, N)
        denom = jnp.sum(df.conj() * df, axis=-1)  # (B, )

        dHn = numer / denom[:, None, None] # (B, N, N)

        norm = jnp.linalg.norm(fx, axis=-1)
        print(jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn) # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps


        return i+1, xnext, fxn, Hnext, norm < eps

    f0 = func(x0)
    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N)),
        jnp.zeros(B, dtype=bool)
    )
    state = init_val
    while cond_fun(state):
        state = body_fun(state)
    # state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux
    return OptimResult(x, fx,i, H)

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


        Hdf = jnp.matmul(H, df[..., None]).squeeze(-1) # (B, N)
        numer = jnp.expand_dims((dx - Hdf), -1) * jnp.expand_dims(df, -2) # (B, N)
        denom = jnp.sum(df.conj() * df, axis=-1)  # (B, )

        dHn = numer / denom[:, None, None] # (B, N, N)

        norm = jnp.linalg.norm(fx, axis=-1)
        # print(jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn) # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps


        return i+1, xnext, fxn, Hnext, norm < eps

    f0 = func(x0)
    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N)),
        jnp.zeros(B, dtype=bool)
    )
    # state = init_val
    # while cond_fun(state):
    #     state = body_fun(state)
    state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux
    return OptimResult(x, fx,i, H)

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

        first_term = jnp.expand_dims(dx, axis=-1) * jnp.expand_dims(dx, axis=-2) / jnp.sum(dx.conj() * df, axis=-1)[:, None, None]

        n_v = jnp.matmul(H, df[..., None]).squeeze(-1)

        second_numerator = jnp.expand_dims(n_v, axis=-1) * jnp.expand_dims(n_v, axis=-2)
        second_denom = jnp.sum(df.conj() * n_v, axis=-1)
        second_term = -second_numerator / second_denom[:, None, None]



        dHn = first_term + second_term

        norm = jnp.linalg.norm(fx, axis=-1)
        print(jnp.mean(norm), jnp.mean(done))

        xnext = jnp.where(done[:, None], x, xn) # Keep old or carry on
        Hnext = jnp.where(done[:, None, None], H, H + dHn)
        # done = norm < eps


        return i+1, xnext, fxn, Hnext, norm < eps

    f0 = func(x0)
    B, N = f0.shape
    init_val = (
        0,
        x0,
        f0,
        jnp.broadcast_to(jnp.eye(N, dtype=f0.dtype)[None], (B, N, N)),
        jnp.zeros(B, dtype=bool)
    )
    state = init_val
    while cond_fun(state):
        state = body_fun(state)
    # state = jax.lax.while_loop(cond_fun, body_fun, init_val)
    i, x, fx, H, done = state
    # Return sol, aux
    return (x, OptimResult(i, fx, H))
    return state
