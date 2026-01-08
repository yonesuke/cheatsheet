# Flax (NNX) Cheat Sheet

> [!WARNING]
> This cheat sheet focuses exclusively on **Flax NNX**.
> **DO NOT USE `flax.linen`**. Linen is the legacy API and has significantly different semantics (implicit state, apply methods, etc.). NNX is the modern, Pythonic standard.

## Core Concepts (NNX)

NNX brings "Normal Python" object-oriented programming to JAX.
*   **Explicit State**: Parameters are stored directly on the object (like PyTorch).
*   **Reference Semantics**: Objects are mutable Python objects. `nnx.jit` handles the magic.

### 1. Defining a Module
Inherit from `nnx.Module`. Define layers in `__init__`, forward pass in `__call__`.
**Crucial**: You must pass `rngs` to `__init__` if your layers need randomness (initialization).

```python
from flax import nnx
import jax.numpy as jnp

class MLP(nnx.Module):
    def __init__(self, in_features, hidden_dim, out_features, *, rngs: nnx.Rngs):
        # Initialize layers with explicit shapes (No shape inference!)
        self.linear1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs) # Rate 0.5

    def __call__(self, x, *, rngs: nnx.Rngs = None):
        x = self.linear1(x)
        x = nnx.relu(x)
        # Pass rngs for stochastic layers like Dropout
        x = self.dropout(x, rngs=rngs) 
        x = self.linear2(x)
        return x

# Initialization
rngs = nnx.Rngs(params=0, dropout=1) # "params" key for weights context
model = MLP(in_features=10, hidden_dim=32, out_features=1, rngs=rngs)

# Forward pass (Eager)
dummy_input = jnp.ones((1, 10))
output = model(dummy_input, rngs=rngs) # Output shape: (1, 1)
```

### 2. Managing Randomness (`nnx.Rngs`)
No more manual `random.split` threading for layers! `nnx.Rngs` handles stream splitting.

```python
# Create a collection of RNG streams
rngs = nnx.Rngs(
    params=42,  # Used for initialization
    dropout=0,  # Used for dropout masks
    noise=123   # Custom stream
)

# Call split() inside layers/models to get a fresh key
key = rngs.dropout() 
```

### 3. Training Loop (`nnx.Optimizer` & `nnx.jit`)
Use `nnx.Optimizer` (wraps Optax) and `nnx.jit`.
`nnx.jit` automatically manages the state update propagation.

```python
import optax

# 1. Setup Model & Optimizer
model = MLP(10, 32, 1, rngs=nnx.Rngs(0))
# wrt=nnx.Param tells the optimizer to only update parameters (ignoring BatchStats etc.)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# 2. Define Train Step (JIT compiled)
@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        # Call model with a dropout 'call' key
        # We create a new Rngs context just for this forward pass
        preds = model(x, rngs=nnx.Rngs(dropout=0)) 
        return jnp.mean((preds - y) ** 2)

    # Gradients w.r.t NNX Parameters
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Update model parameters in-place (conceptually)
    optimizer.update(grads) 
    
    return loss

# 3. Execution
x_batch = jnp.ones((32, 10))
y_batch = jnp.ones((32, 1))

for i in range(100):
    loss = train_step(model, optimizer, x_batch, y_batch)
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss}")
```

### 4. Advanced: Functional API (Interfacing with Raw JAX)
If you need to use raw `jax.lax.scan` or `jax.vmap` deeply, you might want to split the model into execution graph (`GraphDef`) and State.

| Function | Description |
| :--- | :--- |
| `nnx.split(model)` | Returns `(graphdef, state)`. pure Pytrees. |
| `nnx.merge(graphdef, state)` | Reconstructs the `model` object. |
| `nnx.update(model, state)` | Updates the mutable model with new state. |

```python
# Split for pure functional transformation
graphdef, state = nnx.split(model)

@jax.jit
def pure_forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    y = model(x)
    # Extract updated state (e.g. BatchNorm stats)
    _, new_state = nnx.split(model)
    return y, new_state

y, new_state = pure_forward(graphdef, state, x_batch)
nnx.update(model, new_state) # Sync back to object
```

### 5. `nnx.Variable` Types
*   **`nnx.Param`**: Learnable parameters (weights, biases). Optimized by `nnx.Optimizer`.
*   **`nnx.BatchStat`**: BatchNorm statistics (running mean/var). Not optimized, but updated during forward.
*   **`nnx.Cache`**: Autoregressive cache for Transformers.

```python
class MyLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.w = nnx.Param(rngs.params.normal((10, 10)))
        # Custom state variable not touched by optimizer
        self.counter = nnx.Variable(jnp.array(0)) 
```
