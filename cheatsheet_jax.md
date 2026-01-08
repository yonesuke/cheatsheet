# JAX Cheat Sheet for Machine Learning & Analysis

This cheat sheet covers the essential tools for using JAX in machine learning and mathematical analysis.

## Core JAX Concepts

JAX is essentially **NumPy on accelerators (GPU/TPU)** + **Composable function transformations**.

### 1. JAX vs NumPy
JAX arrays (`jax.Array`) are immutable.

```python
import jax
import jax.numpy as jnp
import numpy as np

# Creation (similar to NumPy)
x = jnp.arange(10)
y = jnp.linspace(0, 1, 10)

# Immutable! This will raise an error:
# x[0] = 5  # ❌ generic_error

# Update syntax (returns new array)
x_new = x.at[0].set(5)
```

### 2. The Four Horsemen of JAX (Transformations)

#### `jax.jit`: Just-In-Time Compilation
Compiles your function to XLA for speed. `jit` requires **static shapes**.

```python
@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = jax.random.key(0)
x = jax.random.normal(key, (1000000,))
selu(x).block_until_ready() # Much faster than raw execution
```

#### `jax.grad`: Automatic Differentiation
Computes gradients. By default, differentiates w.r.t 1st argument.

```python
def tanh(x):
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = jax.grad(tanh)

print(grad_tanh(1.0)) # evaluated at x=1.0
```

**`jax.value_and_grad`**: Efficiently return both loss and gradient.
```python
loss_fn = lambda x: x ** 2
loss, grads = jax.value_and_grad(loss_fn)(3.0)
# loss=9.0, grads=6.0
```

#### `jax.vmap`: Auto-Vectorization
Automatically batches operations. Replace generic loops with vectorization.

```python
mat = jax.random.normal(key, (150, 100))
batched_x = jax.random.normal(key, (10, 100))

def apply_matrix(x):
    return jnp.dot(mat, x)  # Works on single vector

# Auto-vectorize over the 0-th dimension of the input
vmap_batched = jax.vmap(apply_matrix)(batched_x)
```

#### `jax.random`: Logic for Random Numbers
JAX randomness is **explicit** and **stateless** (no global seed). You must manage `PRNGKey`.

```python
key = jax.random.key(42)

# Splitting keys (Best Practice)
key, subkey = jax.random.split(key)
random_val = jax.random.normal(subkey, shape=(1,))

# Multi-split
key, *subkeys = jax.random.split(key, num=4)
```

---

## Machine Learning Specifics

### 1. Stateful Computations (The "Params" Pattern)
Since JAX functions must be pure (no side effects), we pass state explicitly.

```python
from typing import NamedTuple

class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray

def init_model(rng, in_dim, out_dim):
    w_key, b_key = jax.random.split(rng)
    return Params(
        weight=jax.random.normal(w_key, (in_dim, out_dim)) * 0.01,
        bias=jax.random.normal(b_key, (out_dim,))
    )

def forward(params: Params, x: jnp.ndarray):
    return jnp.dot(x, params.weight) + params.bias
```

### 2. Loss & Update Loop (Optimization)
A typical raw JAX training step.

```python
@jax.jit
def update_step(params, x, y, learning_rate=0.01):
    def loss_fn(p):
        preds = forward(p, x)
        return jnp.mean((preds - y) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Gradient Descent Update (Pytree map)
    # params - lr * grads
    new_params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
    return new_params, loss
```

### 3. Pytrees
JAX can differentiate through arbitrary nested python structures (lists, tuples, dicts, NamedTuples). `Params` above is a Pytree.

```python
# Flatten a pytree
leaves, treedef = jax.tree.flatten(params)

# Apply function to all leaves
doubled_params = jax.tree.map(lambda x: x * 2, params)
```

---

## Control Flow (Loops & Conditionals)
Use `jax.lax` primitives inside `jit` to keep compilation efficient.

### `jax.lax.cond` (If/Else)
Differentiable branching.
```python
val = jax.lax.cond(
    pred=x > 0,
    true_fun=lambda operand: operand + 1,
    false_fun=lambda operand: operand - 1,
    operand=x
)
```

### `jax.lax.scan` (Efficient Loops)
Use this for RNNs or carrying state through a sequence. compiles a single loop iteration.

```python
def scan_body(carry, x):
    # carry: accumulated state
    # x: input for this step
    new_carry = carry + x
    output = new_carry * 2
    return new_carry, output

init_carry = 0
xs = jnp.array([1, 2, 3])
final_carry, outputs = jax.lax.scan(scan_body, init_carry, xs)
```

### `jax.lax.while_loop`
Standard while loop for JIT-compiled code. `cond_fun` must return a boolean.

```python
def cond_fun(val):
    return val < 10

def body_fun(val):
    return val + 1

init_val = 0
# Equivalent to: while val < 10: val += 1
result = jax.lax.while_loop(cond_fun, body_fun, init_val)
```

### `jax.lax.fori_loop`
A lower-level loop similar to `for i in range(lower, upper)`. Often faster to compile than python `for` loops if loop count is large but fixed.

```python
def body_fun(i, val):
    return val + i

init_val = 0
lower = 0
upper = 10
# Equivalent to: for i in range(0, 10): val += i
result = jax.lax.fori_loop(lower, upper, body_fun, init_val)
```

---

## Parallelism (Sharding)
Modern JAX uses `jax.sharding` for distributed arrays (single-program multi-data).

```python
from jax.sharding import NamedSharding, Mesh
from jax.sharding import PartitionSpec as P

# 1. Define Mesh (e.g., 8 devices)
devices = jax.devices()
# Assumes you have multiple devices available
# mesh = Mesh(devices, axis_names=('data',))

# 2. Define Sharding Spec
# Shard along 'data' axis (batch dimension)
# data_sharding = NamedSharding(mesh, P('data'))

# 3. Create/Move Array with Sharding
# JAX automatically distributes operations on this array
# x_sharded = jax.device_put(x_large, data_sharding)

# 4. Computation (Auto-parallelized)
# ex: result is effectively computed in parallel across devices
# y = jnp.sin(x_sharded)
```

## Useful Tools for Math

*   **`jax.scipy`**: Drop-in replacements for `scipy` functions (e.g., `jax.scipy.optimize`, `jax.scipy.stats`).
*   **`jax.numpy.linalg`**: Optimized linear algebra (SVD, Cholesky, Eigendecomposition).

```python
import jax.scipy.stats as stats

# Differentiable PDF
p = stats.norm.pdf(x, loc=0, scale=1)
```
