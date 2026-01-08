# Optax Cheat Sheet

[Optax](https://github.com/google-deepmind/optax) is a gradient processing and optimization library for JAX. It is designed to be composable and flexible.

> [!NOTE]
> Optax is state-based (functional). Optimizers don't mutate parameters in place; they return updates and a new optimizer state.

## 1. Core Concepts

### Gradient Transformation (`GradientTransformation`)
An optimizer in Optax is a pair of pure functions:
1.  `init(params)`: Returns the initial `opt_state`.
2.  `update(updates, state, params=None)`: Returns `(updates, new_state)`.

### Basic Usage Pattern
```python
import jax.numpy as jnp
import optax

# 1. Define Optimizer
optimizer = optax.adam(learning_rate=1e-3)

# 2. Initialize State
params = {'w': jnp.ones((10,))}
opt_state = optimizer.init(params)

# 3. Update Step (Inside training loop)
grads = {'w': jnp.array([0.1] * 10)} # Computed via jax.grad
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

## 2. Common Optimizers

Most optimizers are available as simple functions.

```python
# Stochastic Gradient Descent
optimizer = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)

# Adam (Standard)
optimizer = optax.adam(learning_rate=1e-3)

# AdamW (Adam with Weight Decay)
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)

# RMSProp
optimizer = optax.rmsprop(learning_rate=1e-3, decay=0.9)

# AdaFactor (Memory efficient, good for Transformers)
optimizer = optax.adafactor(learning_rate=1e-3)

# Lion (Evolved Sign Momentum)
optimizer = optax.lion(learning_rate=1e-4)
```

## 3. Schedules (Learning Rate Decay)

Optax separates schedules from optimizers. A schedule is a function `step -> learning_rate`.

```python
# 1. Create Schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-5
)

# 2. Inject into Optimizer
# Just pass the schedule instead of a float learning_rate
optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
```

**Common Schedules:**
*   `constant_schedule(value)`
*   `cosine_decay_schedule(init_value, decay_steps)`
*   `linear_schedule(init_value, end_value, transition_steps)`
*   `exponential_decay(init_value, decay_rate, transition_steps)`
*   `join_schedules(schedules, boundaries)`: Piecewise schedules.

## 4. Chaining Transformations (`optax.chain`)

You can build custom optimizers by chaining simple transformations.

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip Gradients
    optax.adamw(learning_rate=1e-3, weight_decay=1e-4), # Optimizer
    optax.ema(decay=0.999) # Exponential Moving Average of weights
)
```

**Common Transformations:**
*   `optax.clip(max_delta)`: Clip values.
*   `optax.clip_by_global_norm(max_norm)`: Clip by total norm of gradients.
*   `optax.add_decayed_weights(weight_decay)`: Explicit weight decay.
*   `optax.scale_by_adam()`: Just the scaling part of Adam (no LR).
*   `optax.scale(-learning_rate)`: Apply learning rate (standard SGD update).

## 5. Loss Functions

Optax provides standard loss functions in `optax.losses` (or just `optax`).

```python
# Classification
loss = optax.softmax_cross_entropy(logits, labels) # shape: (batch, num_classes)
loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) # sparse
loss = optax.sigmoid_binary_cross_entropy(logits, labels)

# Regression
loss = optax.l2_loss(predictions, targets) # MSE = 0.5 * (x - y)^2 (sum/mean manually)
loss = optax.huber_loss(predictions, targets, delta=1.0)
loss = optax.cosine_distance(predictions, targets)

# Reducing losses
# Optax losses usually return element-wise loss. You must reduce them.
total_loss = jnp.mean(loss) 
```

## 6. Flax Integration

### A. Flax Linen (`TrainState`)
The standard way to use Optax with Flax Linen is `flax.training.train_state.TrainState`.

```python
from flax.training import train_state
from flax import linen as nn

class TrainState(train_state.TrainState):
    # Extend if you need extended state (e.g. batch stats)
    pass

model = MyModel()
params = model.init(rng, input_data)
tx = optax.adam(1e-3)

# 1. Create TrainState
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# 2. Update Step
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['x'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['y']).mean()
        return loss
    
    grads = jax.grad(loss_fn)(state.params)
    # state.apply_gradients automatically calls tx.update() and optax.apply_updates()
    return state.apply_gradients(grads=grads)
```

### B. Flax NNX (`nnx.Optimizer`)
For the new Flax NNX API, use `nnx.Optimizer`.

```python
from flax import nnx

model = MyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        pred = model(x)
        return optax.l2_loss(pred, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads) # In-place update of model parameters
    return loss
```

## 7. Advanced Tips

### Multi-Optimizer (Masking)
Apply different optimizers to different parameters (e.g., no weight decay for biases).

```python
# 1. Define Mask
def no_decay_mask(params):
    # Return True for parameters that should have weight decay
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: (path[-1] != 'bias') for path in flat_params}
    return flax.traverse_util.unflatten_dict(flat_mask)

# 2. Use masked wrapper or chain with masking
# Modern way: optax.multi_transform
tx = optax.multi_transform(
    {
        'decay': optax.adamw(1e-3, weight_decay=1e-4),
        'no_decay': optax.adam(1e-3),
    },
    param_labels=map_nested_structure_to_labels # You need to map params to 'decay'/'no_decay'
)
```

### Gradient Accumulation
Optax provides a wrapper for gradient accumulation.

```python
optimizer = optax.MultiSteps(
    optax.adam(1e-3),
    every_k_schedule=4 # Accumulate over 4 steps
)
```

## 8. Experimental: Muon Optimizer

[Muon](https://github.com/google-deepmind/optax/blob/main/optax/contrib/muon.py) (Momentumized Newton-Schulz) is a second-order optimizer designed for large-scale training of transformers. It optimizes 2D parameters (weights) using Newton-Schulz iteration while using AdamW for other parameters (biases, embeddings).

**Key Features:**
*   Approximates curvature information efficiently for 2D matrices.
*   Often converges faster than AdamW for large models.
*   Found in `optax.contrib`.

**Usage Pattern:**
Multimodal optimization (Muon for weights, AdamW for others) is required.

```python
import optax
from optax.contrib import muon

def get_muon_optimizer(learning_rate, weight_decay=1e-4):
    # 1. Define Mask for Muon (Only 2D weights, exclude embeddings/biases)
    def is_muon_param(path, param):
        # Example heuristic: Any 2D parameter that is not an embedding
        return param.ndim == 2 and 'embedding' not in str(path).lower()

    # 2. Define Gradient Transformations
    # Muon for 2D weights
    muon_tx = muon.muon(
        learning_rate=learning_rate,
        momentum=0.95,
        nesterov=True,
        ns_steps=5 # Number of Newton-Schulz steps
    )
    
    # AdamW for everything else (biases, layer norms, embeddings)
    adamw_tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # 3. Combine using multi_transform (or partition)
    # Note: partition automatically splits params based on the predicate
    optimizer = optax.partition(
        condition=is_muon_param,
        true_gradient_transform=muon_tx,
        false_gradient_transform=adamw_tx
    )
    
    return optimizer
```
