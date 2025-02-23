import jax
import jax.numpy as jnp
from jax import random

key = random.key(1738)
key1, key2 = random.split(key)
mat = random.normal(key1, (4, 4)) * 0
vec = random.normal(key2, (4, 2, 2))

print(mat)
print(vec)
indices = jnp.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)




print(insert_blocks(mat, indices, vec))