import jax
import jax.numpy as jnp
from typing import NamedTuple

class OptimResult(NamedTuple):
    iterations: int
    fval: jax.Array
    Jinv: jax.Array

def broydenb2(func, x0, eps=1e-10, max_iter=20):
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
    return (x, OptimResult(i, fx, H))
    return state
