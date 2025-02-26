import jax
import jax.numpy as jnp

from jax import custom_jvp

# @custom_jvp
# def f(x, y):
#   return jnp.sin(x) * y

# @f.defjvp
# def f_jvp(primals, tangents):
#   x, y = primals
#   x_dot, y_dot = tangents
#   primal_out = f(x, y)
#   tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
#   return primal_out, tangent_out

@custom_jvp
def f(x):
  return jnp.sin(x) + jnp.cos(x) * jnp.array([1, 2])

@f.defjvp
def f_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  primal_out = f(x)
  tangent_out = (jnp.cos(x) - jnp.sin(x) *  jnp.array([1, 2])) * x_dot
  return primal_out, tangent_out

x0 = jnp.array([1.1, 2.2])
jac = jax.jacfwd(f)

grad = jax.grad(f)
print(f(x0))
print(jac(x0))
print(grad(x0))