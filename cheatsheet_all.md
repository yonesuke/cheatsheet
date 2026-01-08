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


---


# Numerical Interpolation Cheatsheet

This document provides a comprehensive summary of interpolation theories, algorithms, and their implementations in Python libraries (`scipy`, `interpax`).

---

## Taxonomy

Interpolation methods are classified by data dimensionality, arrangement, and desired smoothness.

| Category | Characteristics | Key Methods |
| :--- | :--- | :--- |
| **Global Polynomial** | Passes through all points with a single polynomial | Lagrange, Newton, Barycentric |
| **Piecewise Polynomial** | Uses different polynomials for each interval (Spline) | Linear, Cubic Spline, PCHIP, Akima |
| **Multidimensional (Grid)** | Interpolation on structured grids | Bilinear, Bicubic, RegularGrid |
| **Multidimensional (Scatter)** | Interpolation on unstructured/random points | Nearest, LinearND, Clough-Tocher, RBF |

---

## 1. Global Polynomial Interpolation

For $N+1$ points $(x_0, y_0), \dots, (x_N, y_N)$, a unique polynomial of degree $N$ exists.

### 1.1 Lagrange Interpolation (Barycentric Form)
**Theory**:
The Lagrange polynomial is defined as:
$$ P(x) = \sum_{j=0}^{N} y_j L_j(x), \quad L_j(x) = \prod_{i \neq j} \frac{x - x_i}{x_j - x_i} $$
For numerical stability and efficiency ($O(N)$ evaluation after $O(N^2)$ precomputation), the **Barycentric form** is preferred:
$$ P(x) = \frac{\sum_{j=0}^N \frac{w_j y_j}{x - x_j}}{\sum_{j=0}^N \frac{w_j}{x - x_j}}, \quad w_j = \frac{1}{\prod_{i \neq j} (x_j - x_i)} $$

**Libraries**:
- **Scipy**: `scipy.interpolate.BarycentricInterpolator`
- **Interpax**: N/A

---

## 2. Piecewise Polynomial (Spline) Interpolation - 1D

Connects data points with low-degree polynomials to avoid Runge's phenomenon.

### 2.1 Cubic Spline Interpolation (Detailed)

**Theory & Derivation**:
A Cubic Spline $S(x)$ consists of piecewise cubic polynomials $S_i(x)$ on each interval $[x_i, x_{i+1}]$.
It satisfies $S(x) \in C^2[x_0, x_N]$ (continuous up to 2nd derivative).

**Constraints for $N$ intervals ($4N$ unknowns)**:
1.  **Interpolation**: $S_i(x_i) = y_i, \quad S_i(x_{i+1}) = y_{i+1}$ (Equation count: $2N$)
2.  **Continuity ($C^1$)**: $S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})$ ($N-1$)
3.  **Continuity ($C^2$)**: $S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})$ ($N-1$)
4.  **Boundary Conditions**: 2 conditions needed (e.g., Natural: $S''(x_0)=S''(x_N)=0$).

**Derivation via Moments**:
Let $M_i = S''(x_i)$ be the "moments" (2nd derivatives). Since $S_i(x)$ is cubic, $S''_i(x)$ is linear.
$$ S''_i(x) = \frac{x_{i+1}-x}{h_i}M_i + \frac{x-x_i}{h_i}M_{i+1}, \quad h_i = x_{i+1}-x_i $$
Integrating twice and applying interpolation conditions yields:
$$ S_i(x) = \frac{(x_{i+1}-x)^3 M_i + (x-x_i)^3 M_{i+1}}{6h_i} + \left(y_i - \frac{M_i h_i^2}{6}\right)\frac{x_{i+1}-x}{h_i} + \left(y_{i+1} - \frac{M_{i+1} h_i^2}{6}\right)\frac{x-x_i}{h_i} $$
Differentiating to get $S'_i(x)$ and enforcing $S'_{i-1}(x_i) = S'_i(x_i)$ leads to the fundamental linear system for $M$:
$$ \frac{h_{i-1}}{6}M_{i-1} + \frac{h_{i-1}+h_i}{3}M_i + \frac{h_i}{6}M_{i+1} = \frac{y_{i+1}-y_i}{h_i} - \frac{y_i-y_{i-1}}{h_{i-1}} $$
This forms a **Tridiagonal System** $A\mathbf{M} = \mathbf{d}$ which can be solved in $O(N)$ time.

**Algorithm (Natural Cubic Spline)**:
1.  **Calculate Steps**: $h_i = x_{i+1} - x_i$.
2.  **Build System**: Construct tridiagonal matrix $A$ and RHS vector $d$.
    - Diagonal: $2(h_{i-1} + h_i)$ (scaled) or similar standard form.
    - Off-diagonal: $h_{i-1}, h_i$.
    - RHS: $6 \times$ divided differences of $y$.
3.  **Solve**: Use Thomas Algorithm (TDMA) to find $M_0 \dots M_N$.
4.  **Evaluate**: Use the $S_i(x)$ formula derived above for any query point.

**Pseudo-Code**:
```python
def cubic_spline_1d(x, y, query_points):
    N = len(x) - 1
    h = x[1:] - x[:-1]
    
    # 1. Setup Tridiagonal System A * M = b
    # alpha[i] * M[i-1] + beta[i] * M[i] + gamma[i] * M[i+1] = b[i]
    alpha = h[:-1]
    beta = 2 * (h[:-1] + h[1:])
    gamma = h[1:]
    b = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    
    # Natural Boundary Conditions (M[0]=0, M[N]=0) -> Adjust matrix implicitly
    # In practice, solve the (N-1)x(N-1) inner system
    
    # 2. Solve for Moments M using TDMA (Thomas Algorithm)
    M_inner = tdma_solve(alpha, beta, gamma, b)
    M = [0] + M_inner + [0] # Concatenate boundary values
    
    # 3. Evaluate results
    results = []
    for q in query_points:
        # Find interval i where x[i] <= q < x[i+1]
        i = search_interval(x, q) 
        dx = q - x[i]
        
        # Cubic evaluation form
        # S(x) = a_i + b_i * dx + c_i * dx^2 + d_i * dx^3
        # Coefficients derived from y and M
        a = y[i]
        b = (y[i+1] - y[i])/h[i] - (2*M[i] + M[i+1])*h[i]/6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * h[i])
        
        val = a + b*dx + c*dx**2 + d*dx**3
        results.append(val)
    return results
```

**Libraries**:
- **Scipy**: `CubicSpline`, `make_interp_spline`
- **Interpax**: `interp1d(method='cubic')` (local C1), `method='cubic2'` (natural C2)

---

### 2.2 Other Splines
- **Linear**: Connects points with lines ($C^0$). High speed, low accuracy.
- **PCHIP / Monotonic**: Preserves monotonicity. $C^1$ smooth. No overshoots.
- **Akima**: Visually pleasing, robust to outliers. $C^1$ smooth.

---

## 3. Multidimensional Interpolation - 2D

### 3.1 Bicubic Interpolation (Structured Grid)

**Theory**:
Bicubic interpolation extends cubic splines to 2D regular grids. It approximates the function within a grid square $[x_i, x_{i+1}] \times [y_j, y_{j+1}]$ using the tensor product:
$$ f(x, y) = \sum_{p=0}^3 \sum_{q=0}^3 a_{pq} (x-x_i)^p (y-y_j)^q $$
There are 16 coefficients ($a_{00} \dots a_{33}$). These are determined by matching 16 constraints at the 4 corner points of the cell:
- **Function values**: $f(x, y)$ (4 constraints)
- **Gradient**: $\partial f / \partial x$, $\partial f / \partial y$ (8 constraints)
- **Cross-derivative**: $\partial^2 f / \partial x \partial y$ (4 constraints)

**Global vs. Local**:
- **Global Bicubic Spline**: Solves for derivatives globally across the entire grid (C2 continuity everywhere). This is equivalent to performing 1D cubic spline interpolation along rows, then along columns of the result.
- **Local Bicubic (e.g., Catmull-Rom)**: Estimates derivatives using only local neighbors. Faster but lower continuity.

**Algorithm (Global Tensor Product approach)**:
1.  **Row Interpolation**: For a query point $(x_q, y_q)$, first perform 1D interpolation on every row $y_j$ at $x_q$. This yields temporary values $z_j \approx f(x_q, y_j)$.
2.  **Column Interpolation**: Use the values $z_j$ to perform 1D interpolation along the vertical direction at $y_q$.

**Pseudo-Code (Tensor Product)**:
```python
def bicubic_interp_2d(X_grid, Y_grid, Z_grid, x_q, y_q):
    # X_grid, Y_grid are 1D arrays defining grid lines
    # Z_grid is Matrix of values
    
    # 1. Interpolate along the x-dimension
    # Create a temporary column vector of values at x_q for each row
    z_temp = []
    for row_idx in range(len(Y_grid)):
        # Use 1D Cubic Spline logic here
        val = cubic_spline_1d_single_point(X_grid, Z_grid[row_idx, :], x_q)
        z_temp.append(val)
    
    # 2. Interpolate along the y-dimension
    # Use the temporary values to find value at y_q
    result = cubic_spline_1d_single_point(Y_grid, z_temp, y_q)
    
    return result
```

**Libraries**:
- **Scipy**: `RegularGridInterpolator` (method='cubic'), `RectBivariateSpline`
- **Interpax**: `interp2d(method='cubic')`

---

## 4. Library Methods & Cheat Sheet

| Method Name | Scipy (`scipy.interpolate`) | Interpax (`interpax`) | Smoothness | Best For... |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** | `interp1d(kind='linear')` | `interp1d(method='linear')` | $C^0$ | Speed, simplicity. |
| **Nearest** | `interp1d(kind='nearest')` | `interp1d(method='nearest')` | Discontinuous | Categorical data, avoiding stored values. |
| **Cubic (Spline)**| `CubicSpline` | `interp1d(method='cubic2')` | $C^2$ | Smooth physics/math problems (Natural Spline). |
| **PCHIP/Mono** | `PchipInterpolator` | `interp1d(method='monotonic')` | $C^1$ | Distributions, preventing negative overshoots. |
| **Akima** | `Akima1DInterpolator` | `interp1d(method='akima')` | $C^1$ | Reducing wiggles from outliers. |
| **Grid 2D/3D** | `RegularGridInterpolator` | `interp2d`, `interpolator2d` | Depends | Image processing, field simulation. |
| **Scatter** | `griddata`, `RBFInterpolator` | N/A | Depends | Geo-spatial data, irregular sensor data. |

### Quick Tips
- **Scipy `interp1d` vs `CubicSpline`**: `CubicSpline` is newer, faster, and provides derivative methods. Use it for splines.
- **Interpax**: Use `Interpolator1D` class if you need to interpolate multiple times with the same x-grid (precomputes coefficients).
- **Runge's Phenomenon**: Avoid high-degree global polynomials (Lagrange) on equidistant points. Use Splines or Chebyshev nodes.

---

## 5. Code Examples

### 5.1 1D Interpolation (Splines)

#### Scipy
```python
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

# Data
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 2, 1, 3, 0])
x_new = np.linspace(0, 4, 100)

# Cubic Spline (Natural C2 or C1)
# default implies not-a-knot boundary conditions
cs = CubicSpline(x, y) 
y_cubic = cs(x_new)

# Monotonic Spline (PCHIP) - prevents overshoots
pchip = PchipInterpolator(x, y)
y_mono = pchip(x_new)
```

#### Interpax (JAX)
```python
import jax.numpy as jnp
import interpax

# Data (must be JAX arrays usually)
x = jnp.array([0., 1., 2., 3., 4.])
y = jnp.array([0., 2., 1., 3., 0.])
x_new = jnp.linspace(0, 4, 100)

# Functional API (One-shot)
# method='cubic2' is Natural Cubic Spline (C2)
y_cubic = interpax.interp1d(x_new, x, y, method='cubic2')

# method='monotonic' is PCHIP equivalent
y_mono = interpax.interp1d(x_new, x, y, method='monotonic')

# Interpolator Object (Efficient for repeated calls with same x)
# Precomputes derivatives/coefficients
interpolator = interpax.Interpolator1D(x, y, method='cubic2')
y_cubic_fast = interpolator(x_new)
```

### 5.2 2D Regular Grid Interpolation

#### Scipy
```python
from scipy.interpolate import RegularGridInterpolator

# Grid definition
x = np.linspace(0, 5, 10)
y = np.linspace(0, 5, 10)
xg, yg = np.meshgrid(x, y, indexing='ij')
data = np.sin(xg) * np.cos(yg)

# Define interpolator
# Bounds_error=False and fill_value=None allows extrapolation
interp = RegularGridInterpolator((x, y), data, method='cubic')

# Query
pts = np.array([[1.5, 2.5], [3.1, 4.2]])
vals = interp(pts)
```

#### Interpax (JAX)
```python
import jax.numpy as jnp
import interpax

# Grid definition
x = jnp.linspace(0, 5, 10)
y = jnp.linspace(0, 5, 10)
# Data shape must match (len(x), len(y))
xg, yg = jnp.meshgrid(x, y, indexing='ij')
data = jnp.sin(xg) * jnp.cos(yg)

# Query coordinates (vectorized)
xq = jnp.array([1.5, 3.1])
yq = jnp.array([2.5, 4.2])

# Functional API
# Note: interpax takes separate xq, yq arrays, not a list of points
vals = interpax.interp2d(xq, yq, x, y, data, method='cubic')

# Class API (Faster for fixed grid)
interp_obj = interpax.Interpolator2D(x, y, data, method='cubic')
vals_fast = interp_obj(xq, yq)
```

### 5.3 Scatter Interpolation (Scipy Only)

Interpax currently focuses on grid interpolation. For unstructured data, Scipy is the standard.

#### Scipy
```python
from scipy.interpolate import griddata, RBFInterpolator

# Random scatter points
points = np.random.rand(100, 2) # (N, 2)
values = np.sin(points[:,0]) * np.cos(points[:,1])

# Query grid
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]

# 1. griddata (Method: nearest, linear, cubic)
# Interpolates onto a grid from scatter points
grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

# 2. RBF (Radial Basis Function) - Smoother, works for N-dimensions
rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
# input to RBF needs flattened query points
query_pts = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
rbf_z = rbf(query_pts).reshape(100, 100)
```


---


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


---


# Linear Solver Cheatsheet

This document provides a comprehensive overview of linear equation solving algorithms available in **Julia**, **JAX**, **PyTorch**, **Lineax**, and **NumPy**. It covers the theory, convergence properties, usage scenarios, and implementation details for each library.

## 1. Algorithm Overview

This section covers the theoretical underpinnings of common linear solvers.

### Direct Methods
Direct methods compute the exact solution (up to floating-point error) in a finite number of steps, typically involving matrix factorization.

| Algorithm | Theory | Convergence / Stability | Best For | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **LU (Gaussian Elimination)** | Factors $A = PLU$ (Permutation, Lower, Upper). Reduces problem to two triangular solves. | **Stable** (with partial pivoting). Backward stable. | General square, non-singular matrices. Default for most dense solvers. | $O(N^3)$ (factor) + $O(N^2)$ (solve) |
| **Cholesky** | Factors $A = LL^T$ (Lower triangular). Requires $A$ to be Symmetric and Positive Definite (SPD). | **Very Stable**. Fails if matrix is not PD. ~2x faster than LU. | SPD matrices (e.g., covariance matrices, physics simulations). | $\frac{1}{3}N^3$ |
| **QR Decomposition** | Factors $A = QR$ (Values Orthogonal, Upper Triangular). Solves $Rx = Q^T b$. | **Extremely Stable**. Better numerical properties than LU for ill-conditioned matrices. | Least squares, rectangular systems, or highly ill-conditioned square matrices. | $O(N^3)$ (higher constant than LU) |
| **SVD (Singular Value)** | Factors $A = U \Sigma V^T$. Solves via pseudoinverse $x = V \Sigma^+ U^T b$. | **Most Stable**. Handles rank-deficient and near-singular matrices perfectly. | Rank-deficient systems, minimum-norm least squares, analysis of system stability. | $O(N^3)$ (very high constant) |
| **Diagonal / Triangular** | Direct substitution (Forward/Backward). | **Exact** and stable. | Diagonal or Triangular systems. Often the final step of other factorizations. | $O(N)$ (diag) / $O(N^2)$ (tri) |
| **TDMA (Thomas Algorithm)** | Gaussian elimination optimized for tridiagonal systems. | **Stable** for diagonally dominant or SPD matrices. Unstable otherwise. | 1D PDEs (heat/wave equations), cubic splines, time-series smoothing. | $O(N)$ |

### Iterative Methods (Krylov Subspace)
Iterative methods approximate the solution by minimizing an error function over a subspace. They are preferred for large sparse matrices where $O(N^2)$ storage of factors is prohibitive.

| Algorithm | Theory | Convergence | Best For | Memory |
| :--- | :--- | :--- | :--- | :--- |
| **CG (Conjugate Gradient)** | Minimizes error in $A$-norm over Krylov subspace. | Depends on $\sqrt{\kappa(A)}$ (condition number) and eigenvalue clustering. Guaranteed for SPD. | **Large Sparse SPD matrices**. | Low ($O(N)$) |
| **GMRES** (Generalized Minimal Residual) | Minimizes residual norm $\|b - Ax_k\|_2$. Arnoldi iteration. | Monotonically decreases residual. Depends on eigenvalue distribution. | **General non-symmetric** square systems. | High (stores basis vectors; often restarted: GMRES(k)). |
| **BiCGStab** (Bi-Conjugate Gradient Stabilized) | Variation of BiCG using updates to smooth convergence. | Irregular convergence (spiky residue), but often faster than GMRES per step. No theoretical guarantee. | **General non-symmetric** systems where GMRES memory is too high. | Low ($O(N)$) |

---

## 2. Library Implementations

### Julia (`LinearAlgebra` & `LinearSolve.jl`)
Julia uses a powerful **polyalgorithm** via the `\` operator, dispatching to LAPACK (dense) or SuiteSparse/specialized code (sparse).

*   **Dense `A \ b`**: Checks properties (Triangular -> Diagonal -> Tridiagonal -> Hermitian -> General).
    *   **Tridiagonal**: Optimized $O(N)$ Thomas algorithm (via LAPACK `dgtsv` or native).
    *   **SPD**: Calls LAPACK `dposv` (Cholesky).
    *   **General**: Calls LAPACK `dgsjv` (LU).
    *   **Rectangular**: Calls LAPACK `dgels` (QR min-norm solution).
*   **Sparse `A \ b`**:
    *   **SPD**: CHOLMOD (Cholesky).
    *   **General**: UMFPACK (LU).
*   **Iterative**: Available via packages `IterativeSolvers.jl` or `Krylov.jl`. `LinearSolve.jl` provides a unified interface.

### Lineax (JAX Ecosystem)
[Lineax](https://docs.kidger.site/lineax/) is a dedicated JAX library for linear solves, designed for differentiation and structure awareness.

*   **API**: `lineax.linear_solve(operator, vector, solver=...)`
*   **Solvers**:
    *   `lineax.AutoLinearSolver`: Automatically selects based on operator structure (e.g., `TridiagonalLinearOperator` $\to$ `Tridiagonal`, `DiagonalLinearOperator` $\to$ `Diagonal`, `MatrixLinearOperator` $\to$ `LU` or `QR`).
    *   `lineax.Tridiagonal`: $O(N)$ solver for tridiagonal operators.
    *   `lineax.LU`, `lineax.QR`, `lineax.SVD`: Standard direct solvers.
    *   `lineax.Cholesky`: For PD operators.
    *   `lineax.CG`, `lineax.GMRES`, `lineax.BiCGStab`: Iterative solvers written in JAX.
*   **Specialty**: Fully differentiable, works with diffrax (ODEs), supports PyTrees.

### JAX (`jax.numpy` & `jax.scipy`)
JAX wraps standard LAPACK/cuSOLVER routines.

*   **Dense**: `jax.numpy.linalg.solve` (LU), `jax.numpy.linalg.lstsq` (SVD/QR).
    *   *Note*: On GPU, this uses cuSOLVER.
*   **Sparse**: `jax.scipy.sparse.linalg` contains **iterative** solvers only (`cg`, `gmres`, `bicgstab`).
    *   *Note*: JAX has very limited direct sparse solver support (experimental `spsolve` exists but is limited).
*   **Tridiagonal**: `jax.lax.linalg.tridiagonal_solve` (TPU tailored, uses Thomas Algorithm).

### PyTorch (`torch.linalg`)
PyTorch provides dense solvers similar to NumPy/JAX, powered by MAGMA/cuSOLVER on GPU.

*   **Dense**: `torch.linalg.solve` (LU), `torch.linalg.lstsq` (QR/SVD).
*   **Sparse**: Limited direct support. `torch.sparse` exists but solving systems usually requires conversion to dense or external libraries, though simple sparse-dense solves exist.

### NumPy (`numpy.linalg`)
The standard CPU reference.

*   **Dense**: `numpy.linalg.solve` (LAPACK `_gesv` LU).
*   **Tridiagonal**: Use `scipy.linalg.solve_banded` (LAPACK `dgbsv`) for $O(N)$ performance.
*   **Sparse**: Does **not** exist in `numpy`. Users must use `scipy.sparse.linalg`.

---

## 3. Quick Comparison Table

| Feature | Julia | Lineax (JAX) | JAX (Native) | PyTorch | NumPy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Default (`\`)** | Polyalgorithm (LU/QR/Chol) | `AutoLinearSolver` | LU (`solve`) | LU (`solve`) | LU (`solve`) |
| **Sparse Direct** | **Yes** (integrated) | No (dense mostly) | No (experimental) | No | No (use SciPy) |
| **Sparse Iterative**| Via Packages | **Yes** (CG/GMRES) | **Yes** (CG/GMRES) | Limited | No (use SciPy) |
| **Gradients** | Via Zygote/Enzyme | **First-class** | First-class | First-class | No |
| **GPU** | Yes (CUDA.jl) | Yes | Yes | Yes | No |

---

## 4. Sample Code (Cheatsheet)

### NumPy (CPU, Dense)
```python
import numpy as np

# System: Ax = b
N = 100
A = np.random.rand(N, N)
# Ensure non-singularity
A = A + np.eye(N) * N
b = np.random.rand(N)

# 1. Standard Solve (LU)
x_lu = np.linalg.solve(A, b)

# 2. Least Squares (if A is not square)
A_rect = np.random.rand(N + 10, N)
b_rect = np.random.rand(N + 10)
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_rect, b_rect, rcond=None)

# 3. Cholesky (Manual)
# Only for SPD matrices: A must be symmetric and positive definite
A_spd = A @ A.T  # Construct SPD matrix
L = np.linalg.cholesky(A_spd) # A = L L^T
# Solve L y = b
y = np.linalg.solve(L, b) 
# Solve L^T x = y
x_chol = np.linalg.solve(L.T, y)
```

### JAX (GPU/TPU, Dense & Sparse Iterative)
```python
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg

key = jax.random.PRNGKey(0)

# Data Generation
N = 100
A = jax.random.normal(key, (N, N))
A = A + jnp.eye(N) * N # Diagonally dominant -> non-singular
b = jax.random.normal(key, (N,))

# 1. Standard Dense Solve (LU on GPU)
x_dense = jnp.linalg.solve(A, b)

# 2. Sparse / Iterative Solve (CG)
# JAX requires a linear operator function (matvec)
def matvec(x):
    return A @ x

# CG requires Symmetric Positive Definite (SPD) matrix usually
A_spd = A.T @ A
b_spd = A.T @ b
# Solving (A^T A) x = A^T b which is the normal equation
x_cg, info = jax.scipy.sparse.linalg.cg(
    lambda v: A_spd @ v, 
    b_spd, 
    maxiter=1000
)

# 3. GMRES (General matrices)
x_gmres, info = jax.scipy.sparse.linalg.gmres(matvec, b)
```

### PyTorch (GPU, Autograd)
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 100
A = torch.randn(N, N, device=device)
A = A + torch.eye(N, device=device) * N
b = torch.randn(N, 1, device=device) # Must be (N, 1) or (N,) depending on version

# 1. Standard Solve (LU)
x_lu = torch.linalg.solve(A, b)

# 2. Cholesky Solve
# Faster for SPD systems
A_spd = A @ A.T
# torch.linalg.cholesky_ex is safer for avoiding errors
L = torch.linalg.cholesky(A_spd)
x_chol = torch.cholesky_solve(b, L) # Takes L, not A

# 3. Least Squares
# driver='gels' (QR) or 'gelsd' (SVD - more stable)
x_lstsq, residuals, rank, singular_values = torch.linalg.lstsq(A, b, driver='gels')
```

### Lineax (Advanced JAX Solvers)
```python
import jax
import jax.numpy as jnp
import lineax as lx

key = jax.random.PRNGKey(0)
N = 100
A_val = jax.random.normal(key, (N, N))
b = jax.random.normal(key, (N,))

# Lineax uses "LinearOperators"
operator = lx.MatrixLinearOperator(A_val)

# 1. Auto Solve (Best Practice)
# Automatically chooses LU, QR, etc.
solver = lx.AutoLinearSolver(well_posed=True)
solution = lx.linear_solve(operator, b, solver)
print(solution.value)

# 2. Explicit Choice (e.g., QR)
solver_qr = lx.QR()
solution_qr = lx.linear_solve(operator, b, solver_qr)


# ... (previous code)

# 3. Large Scale / Iterative (GMRES)
# Useful if 'operator' is defined functionally, not as a matrix
solver_gmres = lx.GMRES(rtol=1e-5, atol=1e-5)
solution_iter = lx.linear_solve(operator, b, solver_gmres)
```

### Special: Tridiagonal Systems (TDMA)
Solving for tridiagonal matrices ($O(N)$).

#### SciPy (Standard Python)
```python
import numpy as np
import scipy.linalg

N = 100
# Banded format: [Upper diag, Main diag, Lower diag]
# shape (3, N) - padding required strictly speaking but usually handled by offsets
# scipy.linalg.solve_banded expects shape (u+l+1, N)
# For tridiagonal (l=1, u=1), shape is (3, N).
# Row 0: upper diagonal (first element ignored)
# Row 1: main diagonal
# Row 2: lower diagonal (last element ignored)
ab = np.zeros((3, N))
ab[0, 1:] = np.random.rand(N-1) # Upper
ab[1, :] = np.random.rand(N) + 2 # Main (diagonally dominant)
ab[2, :-1] = np.random.rand(N-1) # Lower
b = np.random.rand(N)

x = scipy.linalg.solve_banded((1, 1), ab, b)
```

#### Lineax (JAX)
```python
import jax.random as jr
import lineax as lx
import jax.numpy as jnp

N = 100
diagonal = jr.normal(jr.PRNGKey(0), (N,))
upper = jr.normal(jr.PRNGKey(1), (N-1,))
lower = jr.normal(jr.PRNGKey(2), (N-1,))
b = jr.normal(jr.PRNGKey(3), (N,))

operator = lx.TridiagonalLinearOperator(diagonal, lower, upper)
solution = lx.linear_solve(operator, b, lx.Tridiagonal())
```

#### JAX (Native)
```python
import jax
import jax.numpy as jnp

N = 100
# Inputs must be (B, N) or (N,)
# diagonals format: (3, N) -> [lower, diagonal, upper]
# Note: This is different from scipy's [upper, diagonal, lower] or [upper, diagonal, lower]
dl = jax.random.normal(jax.random.key(0), (N,))
d  = jax.random.normal(jax.random.key(1), (N,))
du = jax.random.normal(jax.random.key(2), (N,))
b  = jax.random.normal(jax.random.key(3), (N,))

# jax.lax.linalg.tridiagonal_solve(dl, d, du, b)
# dl: lower diagonal (first element ignored)
# d:  main diagonal
# du: upper diagonal (last element ignored)
x = jax.lax.linalg.tridiagonal_solve(dl, d, du, b)
```

## 5. Algorithm Details

This section provides a deep dive into the theory, mathematical formulation, and pseudo-algorithms for the solvers mentioned above.

### 5.1 Direct Solvers

Direct solvers factorize matrix $A$ into simpler forms (triangular, diagonal, orthogonal) to make solving $Ax = b$ trivial (e.g., via simple back-substitution).

#### LU Decomposition (Gaussian Elimination)
**Theory**:
Any square matrix $A$ can be decomposed into a lower triangular matrix $L$ (with unit diagonal) and an upper triangular matrix $U$, such that $PA = LU$, where $P$ is a permutation matrix to ensure numerical stability (partial pivoting).
Solving $Ax = b$ becomes:
1.  Solve $Ly = P b$ (Forward substitution)
2.  Solve $Ux = y$ (Backward substitution)

**Pseudo-Code (Simplified without pivoting)**:
```python
function LU_Decomposition(A):
    n = size(A, 1)
    L = eye(n)
    U = copy(A)
    for k = 1 to n-1:
        for i = k+1 to n:
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]
    return L, U
```

**Background**:
This is the standard "Gaussian Elimination" taught in linear algebra. Without pivoting ($P$), it is unstable if diagonal elements are near zero. With partial pivoting ($O(N^3)$), it is the industry standard for general dense systems.

#### Cholesky Decomposition
**Theory**:
If $A$ is Symmetric and Positive Definite (SPD) (i.e., $x^T Ax > 0$ for all $x \neq 0$), it can be factored uniquely as $A = LL^T$, where $L$ is lower triangular.
Solving $Ax=b \implies LL^T x = b$:
1.  Solve $Ly = b$ (Forward)
2.  Solve $L^T x = y$ (Backward)

**Pseudo-Code**:
```python
function Cholesky(A):
    n = size(A, 1)
    L = zeros(n, n)
    for i = 1 to n:
        for j = 1 to i:
            sum_val = sum(L[i, k] * L[j, k] for k = 1 to j-1)
            if i == j: # Diagonal elements
                L[i, j] = sqrt(A[i, i] - sum_val)
            else:
                L[i, j] = (1.0 / L[j, j]) * (A[i, j] - sum_val)
    return L
```

**Background**:
Cholesky is roughly twice as fast as LU because it exploits symmetry (only calculates lower triangle). It is numerically very stable; if the algorithm encounters a negative number inside the square root, it proves the matrix is not positive definite.

#### QR Decomposition
**Theory**:
Factors $A = QR$, where $Q$ is an orthogonal matrix ($Q^T Q = I$) and $R$ is upper triangular.
Solving $Ax = b \implies QRx = b \implies Rx = Q^T b$.
Since $Q$ is orthogonal, multiplying by $Q^T$ does not amplify errors, making it extremely stable.

**Pseudo-Code (Householder Reflections)**:
```python
function QR_Householder(A):
    m, n = size(A)
    Q = eye(m)
    R = copy(A)
    for k = 1 to n:
        x = R[k:m, k]
        # Construct Householder vector v to zero out elements below diagonal
        e1 = zeros(length(x)); e1[0] = 1
        v = sign(x[0]) * norm(x) * e1 + x
        v = v / norm(v)
        # Apply reflection to R and Q
        R[k:m, k:n] = R[k:m, k:n] - 2 * outer(v, dot(v, R[k:m, k:n]))
        Q[k:m, :] = Q[k:m, :] - 2 * outer(v, dot(v, Q[k:m, :]))
    return Q.T, R
```

**Background**:
While Gram-Schmidt is intuitively simpler, Householder reflections are implemented in libraries like LAPACK because they maintain orthogonality much better in floating-point arithmetic.

#### TDMA (Thomas Algorithm)
**Theory**:
A specialized version of Gaussian Elimination for tridiagonal matrices. Since most elements are zero, we only eliminate the sub-diagonal.
System: $a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$

**Pseudo-Code**:
```python
function TDMA(a, b, c, d):
    n = length(d)
    # Forward elimination
    c'[0] = c[0] / b[0]
    d'[0] = d[0] / b[0]
    for i = 1 to n-1:
        temp = b[i] - a[i] * c'[i-1]
        c'[i] = c[i] / temp
        d'[i] = (d[i] - a[i] * d'[i-1]) / temp
    
    # Backward substitution
    x[n-1] = d'[n-1]
    for i = n-2 down to 0:
        x[i] = d'[i] - c'[i] * x[i+1]
    return x
```

**Background**:
This is an $O(N)$ algorithm, essential for 1D PDE solvers (like solving the heat equation implicitly). It is stable if the matrix is strictly diagonally dominant ($|b_i| > |a_i| + |c_i|$).

---

### 5.2 Iterative Solvers (Krylov Subspace)

Direct solvers convert $A$ to a soluble form. Iterative solvers strictly use matrix-vector multiplication ($v \to Av$) to search for the solution in a "Krylov Subspace" $\mathcal{K}_k = \text{span}\{r_0, Ar_0, A^2r_0, \dots\}$.

#### Conjugate Gradient (CG)
**Theory**:
Discovers the solution by generating a sequence of $A$-orthogonal search directions $p_k$. This means $p_i^T A p_j = 0$ for $i \neq j$. This "conjugacy" property ensures that each step brings us optimally closer to the solution in the underlying norm, without undoing previous progress. **Strictly requires $A$ to be SPD.**

**Pseudo-Code**:
```python
function CG(A, b, x0):
    r = b - A @ x0
    p = r
    rho = dot(r, r)
    for k = 1 to max_iter:
        Ap = A @ p
        alpha = rho / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rho = dot(r, r)
        if sqrt(new_rho) < tolerance:
            break
        beta = new_rho / rho
        p = r + beta * p
        rho = new_rho
    return x
```

**Background**:
The magic of CG is that it only requires storage of a few vectors ($x, r, p$). For SPD matrices, it is the theoretical optimal Krylov solver.

#### GMRES (Generalized Minimal Residual)
**Theory**:
For general non-symmetric matrices, we cannot rely on short recurrences like CG. GMRES builds an orthonormal basis for the Krylov subspace explicitly (with Arnoldi iteration) and finds the vector $x_k$ in that subspace that minimizes the residual norm $\| b - Ax_k \|_2$.

**Pseudo-Code (Simplified Arnoldi-based)**:
```python
function GMRES(A, b, x0, m):
    # m is restart parameter (GMRES(m))
    r0 = b - A @ x0
    beta = norm(r0)
    V = [r0 / beta] # Basis vectors
    H = zeros(m+1, m) # Hessenberg matrix
    
    for j = 0 to m-1:
        w = A @ V[j]
        # Arnoldi Orthogonalization (Gram-Schmidt on Krylov vectors)
        for i = 0 to j:
            H[i, j] = dot(w, V[i])
            w = w - H[i, j] * V[i]
        H[j+1, j] = norm(w)
        V.append(w / H[j+1, j])
        
        # Solve least squares for y_k: min || beta * e1 - H_k * y ||
        # Update x = x0 + V_k * y_k
```

**Background**:
GMRES stores all basis vectors $V_k$, so memory grows linearly with iterations. To manage this, we use "Restarted GMRES(m)", where we discard the basis and restart with the current $x$ as guess after $m$ steps. It is the robust default for non-symmetric systems.

#### BiCGStab (Bi-Conjugate Gradient Stabilized)
**Theory**:
An attempt to get the low memory of CG for non-symmetric systems. It uses a "shadow" Krylov subspace (using $A^T$) to maintain short recurrences (biorthogonality) like BiCG, but "stabilizes" the irregular convergence of BiCG by combining it with GMRES-like local minimization steps.

**Pseudo-Code**:
```python
function BiCGStab(A, b, x0):
    r = b - A @ x0
    r_hat = r # Shadow residual, arbitrary
    p = r
    rho = dot(r_hat, r)
    
    for k = 1 to max_iter:
        v = A @ p
        alpha = rho / dot(r_hat, v)
        s = r - alpha * v
        t = A @ s
        omega = dot(t, s) / dot(t, t)
        
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        if norm(r) < tolerance: break
        
        new_rho = dot(r_hat, r)
        beta = (new_rho / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        rho = new_rho
    return x
```

**Background**:
BiCGStab is very popular because it often converges smoothly like GMRES but with constant low memory usage like CG. However, it can "break down" (divide by zero) in rare unlucky cases, unlike GMRES which is robust.



---


# MLflow Cheatsheet

A guide to using MLflow's Python client "raw" (without framework integrations) for maximum control, including detailed post-experiment analysis.

> [!TIP]
> A runnable example script demonstrating these concepts (including a realistic training loop and analysis) is available at [cheatsheet_mlflow_snippets.py](file:///home/yoneda/cheatsheet/cheatsheet_mlflow_snippets.py). You can run it directly with `uv run cheatsheet_mlflow_snippets.py`.

## Setup

```python
import mlflow

# Set the tracking URI to a local directory or a server
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or "file:///path/to/mlruns"
mlflow.set_experiment("my_experiment_name")
```

## Raw Tracking API

Avoiding `mlflow.autolog()` for explicit control over what gets logged.

### Basic Run Context

```python
import mlflow

# Start a run. Using 'with' matches the run lifecycle to the block scope.
with mlflow.start_run(run_name="manual_run_v1") as run:
    
    # 1. Log Parameters (Configuration)
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam"
    })
    
    # 2. Log Metrics (Looping)
    for epoch in range(10):
        # ... training code ...
        loss = 0.5 * (0.9 ** epoch) # Dummy loss
        accuracy = 0.5 + (0.05 * epoch)
        
        # step is optional but essential for time-series plots in UI
        mlflow.log_metrics({"train_loss": loss, "val_acc": accuracy}, step=epoch)
        
    # 3. Log Artifacts (Files)
    with open("output.txt", "w") as f:
        f.write("Some result data")
    
    # Log a local file to the run's artifact store
    mlflow.log_artifact("output.txt", artifact_path="results")
    
    # Log a model (if you want to use pyfunc or just save the pickle manually)
    # mlflow.sklearn.log_model(model, "model") # Integrated way (avoid if strictly "raw")
    # Raw way:
    # with open("model.pkl", "wb") as f: pickle.dump(model, f)
    # mlflow.log_artifact("model.pkl", "models")
```

### Nested Runs

Useful for hyperparameter search or cross-validation within a "parent" experiment run.

```python
with mlflow.start_run(run_name="hyperparam_search_parent") as parent_run:
    for lr in [0.01, 0.001]:
        with mlflow.start_run(run_name=f"trial_lr_{lr}", nested=True) as child_run:
            mlflow.log_param("lr", lr)
            # ... train and log metrics ...
```

## Post-Experiment Analysis & Retrieval

This is capable of finding the "best" run programmatically after experiments effectively acting as a model registry/leaderboard query.

### Searching Runs

`mlflow.search_runs` returns a Pandas `DataFrame`.

```python
import mlflow

# Get all runs from the experiment
experiment_id = mlflow.get_experiment_by_name("my_experiment_name").experiment_id

# Returns a Pandas DataFrame
df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="params.optimizer = 'adam' AND metrics.val_acc > 0.8", # SQL-like filtering
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    order_by=["metrics.val_acc DESC"] # Sort by metric to find best
)

print(df[["run_id", "params.learning_rate", "metrics.val_acc"]].head())
```

### Fetching the Best Run Automatically

Pattern to get the run ID and artifacts of the absolute best model.

```python
# 1. Search and Sort
best_run_df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.train_loss ASC"], # or "metrics.val_acc DESC"
    max_results=1
)

if not best_run_df.empty:
    best_run = best_run_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_loss = best_run["metrics.train_loss"]
    
    print(f"Best Run ID: {best_run_id}, Loss: {best_loss}")
    
    # 2. Retrieve Artifacts from that run
    # If you logged a file "config.json" or "model.pkl"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id, 
        artifact_path="results/output.txt"
    )
    print(f"Artifact downloaded to: {local_path}")
    
    # Validating the content
    with open(local_path, "r") as f:
        print(f.read())
else:
    print("No runs found.")
```

### Useful Filter Strings

| Requirement | Filter String |
| :--- | :--- |
| Specific Param | `params.model_type = 'cnn'` |
| Metric Threshold | `metrics.accuracy >= 0.95` |
| Tag check | `tags.release_candidate = 'true'` |
| Combination | `params.lr < 0.01 AND metrics.loss < 0.1` |


---


# Neural Network Cheatsheet

A comprehensive guide to the fundamental concepts, architectures, and theoretical foundations of Neural Networks and Deep Learning.

---

## 1. Activation Functions

Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. Without them, a multi-layer neural network would be mathematically equivalent to a single linear transformation (a generalized linear model).

### Common Functions

*   **Sigmoid**:
    $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
    *   **Range**: $(0, 1)$
    *   **Derivative**: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
    *   **Pros**: Interpretable as probability.
    *   **Cons**:
        *   **Vanishing Gradient**: For large $|x|$, $\sigma'(x) \approx 0$.
        *   **Non-zero Centered**: Outputs are always positive, which can introduce zig-zagging dynamics in gradient updates.

*   **Tanh (Hyperbolic Tangent)**:
    $$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
    *   **Range**: $(-1, 1)$
    *   **Derivative**: $\tanh'(x) = 1 - \tanh^2(x)$
    *   **Pros**: Zero-centered outputs solve the zig-zagging problem.
    *   **Cons**: Still suffers from vanishing gradient problem as $|x| \to \infty$.

*   **ReLU (Rectified Linear Unit)**:
    $$ f(x) = \max(0, x) $$
    *   **Derivative**: $f'(x) = 1 \text{ if } x > 0 \text{ else } 0$
    *   **Pros**: **Sparsity** and **No Vanishing Gradient** in the positive domain.
    *   **Cons**: "Dying ReLU" problem (dead neurons).

*   **GELU (Gaussian Error Linear Unit)**:
    $$ \text{GELU}(x) = x \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] $$
    *   **Approximation**: $0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$
    *   **Theory**: Can be viewed as a smooth approximation to ReLU, or as "Dropout" applied to the input where the drop probability depends on the input value itself.
    *   **Pros**: Smoothness allows for better optimization landscapes compared to ReLU. Used in BERT, GPT-2/3, ViT.

*   **Swish / SiLU (Sigmoid Linear Unit)**:
    $$ \text{Swish}(x) = x \cdot \sigma(\beta x) $$
    *   **SiLU**: Special case where $\beta=1$.
    *   **Derivative**: $f'(x) = f(x) + \sigma(x)(1 - f(x))$.
    *   **Pros**: Non-monotonic bump for $x < 0$ allows small negative gradients to flow, potentially helping "unstick" dead neurons. Discovered via Neural Architecture Search.

*   **Softmax**:
    $$ \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}} $$
    *   **Usage**: Output layer for multi-class classification.

---

## 2. MLP (Multi-Layer Perceptron)

### Architecture
An MLP consists of an input layer, one or more hidden layers, and an output layer.
$$ h^{(l)} = \phi(W^{(l)}h^{(l-1)} + b^{(l)}) $$

### Theoretical Background: Universal Approximation
*   **Universal Approximation Theorem (Cybenko, 1989; Hornik et al., 1991)**:
    A feedforward network with a **single hidden layer** of finite width can approximate any continuous function $f: [0, 1]^n \to \mathbb{R}$ to arbitrary precision $\epsilon$, given a non-polynomial activation function $\sigma$.
    $$ F(x) = \sum_{i=1}^N v_i \sigma(w_i^T x + b_i) $$
    *   **Note**: Describes *representational capacity*, not *learnability*. The required width $N$ may be exponential in dimension $n$.

### Regularization: Dropout (Srivastava et al., 2014)
A technique to prevent overfitting by randomly dropping units during training.

*   **Logic (Inverted Dropout)**:
    *   **Training**: Randomly set neurons to zero with probability $p$. Mask vector $r \sim \text{Bernoulli}(1-p)$.
        $$ h^{(l)} = \text{Activation}(W h^{(l-1)} + b) \odot \frac{r}{1-p} $$
        scaling by $\frac{1}{1-p}$ maintains the expected magnitude of the activations.
    *   **Inference**: Use the full network without dropout. The scaling during training ensures no scaling is needed at test time.

*   **Theoretical Background**:
    1.  **Ensemble Interpretation**: Dropout can be viewed as training an ensemble of $2^N$ exponentially many thinned subnetworks that share weights. Test time prediction approximates averaging the predictions of these exponentially many models.
    2.  **Bayesian Approximation (Gal & Ghahramani, 2016)**: Dropout training is mathematically equivalent to approximate Variational Inference in deep Gaussian Processes. Enabling dropout at inference time (**MC Dropout**) allows sampling from the approximate posterior distribution, providing a measure of **epistemic uncertainty** without changing the architecture.
    3.  **Co-adaptation**: Prevents complex co-adaptations in which a feature detector is only helpful in the context of several other specific feature detectors.

---

---

## 3. Normalization Layers

### Batch Normalization (BatchNorm)
Standard in CNNs. Normalizes activations across the **mini-batch** dimension.
$$ \hat{x}_{i} = \frac{x_{i} - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} $$
$$ y_{i} = \gamma \hat{x}_{i} + \beta $$
*   **Stats**: $\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}$ computed per channel across batch $(N, H, W)$.
*   **Inference**: Uses running averages of $\mu, \sigma$ tracked during training.
*   **Theory**:
    *   **Internal Covariate Shift**: Originally claimed to reduce the shifting distribution of internal layers.
    *   **Landscape Smoothing**: Later shown to make the optimization landscape significantly smoother (Lipschitzness), allowing larger learning rates (Santurkar et al., 2018).
*   **Cons**: Breaks independence between samples; performance degrades with small batch sizes; problematic for RNNs.

### Layer Normalization (LayerNorm)
Standard in NLP/Transformers. Normalizes across the **feature** dimension for a single sample.
$$ \mu_L = \frac{1}{H} \sum_{j=1}^H x_{ij}, \quad \sigma_L^2 = \frac{1}{H} \sum_{j=1}^H (x_{ij} - \mu_L)^2 $$
$$ \hat{x}_{ij} = \frac{x_{ij} - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} $$
*   **Independence**: Computation is independent of batch size and other samples.
*   **Theory**: Invariance to re-scaling of weight matrix and input signal.

### RMS Normalization (RMSNorm)
Simplified LayerNorm used in modern LLMs (e.g., Llama, Gopher).
$$ \bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{H} \sum_{j=1}^H x_j^2}} \cdot \gamma_i $$
*   **Difference**: Removed mean centering ($\mu_L$). Only re-scales based on Root Mean Square.
*   **Theory**: Hypothesizes that the re-centering invariance of LayerNorm is less important than re-scaling invariance. Reducing operations speeds up training without performance loss.

---

## 4. CNN (Convolutional Neural Networks)

### Architecture
*   **Convolution**: Cross-correlation between input $I$ and kernel $K$.
    $$ (I * K)(x) = \int I(x')K(x-x') dx' $$
*   **Pooling**: Subsampling operation (Max, Average) for dimensionality reduction and invariance.

### Theoretical Background: Equivariance & Inductive Bias
*   **Translation Equivariance**:
    A map $f$ is equivariant to group action $g$ if $f(g \cdot x) = g \cdot f(x)$. Convolutions are equivariant to translation: shifting the input shifts the feature map.
    $$ [\text{Shift}_s(f) * g](t) = \text{Shift}_s(f * g)(t) $$
*   **Priors**: CNNs impose strong priors:
    1.  **Locality**: Features are locally correlated.
    2.  **Stationarity**: Features (edges, textures) are useful everywhere in the image (parameter sharing).

---

## 5. ResNet (Residual Networks)

### Architecture
Introduced to allow training of extremely deep networks (hundreds/thousands of layers) by learning residual functions.
*   **Residual Block**:
    $$ x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W}_l) $$
    where $x_l$ is the input and $\mathcal{F}$ is the residual mapping (e.g., 2 or 3 conv layers).

### Mathematical Theory: The Gradient Highway
ResNet solves the **Vanishing Gradient** degradation problem not just by initialization, but by structure.

1.  **Forward Propagation**:
    By unrolling the recursion $x_{l+1} = x_l + \mathcal{F}(x_l)$, we can express $x_L$ as the sum of $x_l$ plus all intermediate residuals:
    $$ x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) $$
    This contrasts with plain networks $x_L = \prod W_i x_0$, where multiplicative dynamics cause instability.

2.  **Backward Propagation (Gradient Flow)**:
    Consider the gradient of the loss $\mathcal{L}$ with respect to the input of the $l$-th layer $x_l$. Using the chain rule:
    $$ \frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} $$
    From the additive formulation above, the term $\frac{\partial x_L}{\partial x_l}$ becomes:
    $$ \frac{\partial x_L}{\partial x_l} = \frac{\partial}{\partial x_l} \left( x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right) = I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) $$
    Combining these:
    $$ \frac{\partial \mathcal{L}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial x_L}}_{\text{Direct Term}} + \underbrace{\frac{\partial \mathcal{L}}{\partial x_L} \left( \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right)}_{\text{Residual Term}} $$

3.  **Implication**:
    *   The **"Direct Term"** ($\frac{\partial \mathcal{L}}{\partial x_L}$) propagates information directly from the loss to any layer $l$ without passing through the weights of the intermediate layers. The gradient flow is theoretically unimpeded (it does not vanish even if weights are small).
    *   This structure acts as a **Gradient Highway**, allowing gradients to flow effortlessly to earlier layers.
    *   Even if the weights in $\mathcal{F}$ are initialized to be small, the total gradient is dominated by the identity term $I$, ensuring learning can start effectively.

---

## 6. Neural ODEs

### Concept
Taking the layer depth limit $L \to \infty$ and step size $\Delta t \to 0$ of a ResNet leads to a continuous-depth model defined by an ODE.
$$ \frac{dz(t)}{dt} = f(z(t), t, \theta) $$
The output is the solution to the IVP (Initial Value Problem) at time $T$:
$$ z(T) = z(0) + \int_0^T f(z(t), t, \theta) dt $$

### Theoretical Background: Adjoint Sensitivity Method
Standard backpropagation would require storing all intermediate activations, leading to high memory cost. Neural ODEs use the **Adjoint Sensitivity Method** to compute gradients with constant memory cost $O(1)$.
*   **Adjoint State**: Define $a(t) = \frac{\partial \mathcal{L}}{\partial z(t)}$. Its dynamics are given by another ODE running backwards in time:
    $$ \frac{da(t)}{dt} = -a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial z} $$
*   **Gradient Computation**:
    $$ \frac{d\mathcal{L}}{d\theta} = -\int_T^0 a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt $$

---

## 7. RNN, LSTM, GRU & Seq2Seq

### RNN (Recurrent Neural Networks)
Processes sequence data $x_1, \dots, x_T$ by maintaining a hidden state $h_t$ that acts as memory.

*   **Forward Dynamics**:
    $$ h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$
    $$ y_t = W_{hy}h_t + b_y $$

*   **BPTT (Backpropagation Through Time)**:
    Training involves unrolling the network over $T$ time steps. The total loss is $L = \sum_{t=1}^T L_t$.
    Gradients are accumulated backward from $t=T$ to $1$.

*   **Theoretical Issue: Vanishing/Exploding Gradients**:
    Detailed analysis of the gradient flow:
    $$ \frac{\partial h_T}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^T W_{hh}^T \text{diag}(\tanh'(z_t)) $$
    Since $\|\tanh'\| \le 1$, if the largest singular value of $W_{hh}$ is $< 1$, the gradient shrinks exponentially ($a^T \to 0$), making it impossible to learn long-term dependencies. If $> 1$, it explodes ($a^T \to \infty$).

*   **Turing Completeness**: RNNs are **Turing Complete** (Siegelmann & Sontag, 1995). With infinite precision and time, they can simulate any algorithm.

### LSTM (Long Short-Term Memory)
Designed to solve the vanishing gradient problem by introducing a **Cell State** $C_t$ that allows gradients to flow unchanged.

*   **Architecture (Gates)**:
    1.  **Forget Gate**: What to discard from old memory?
        $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
    2.  **Input Gate**: What new information to store?
        $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
        $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad (\text{Candidate update}) $$
    3.  **Cell Update**: Additive interaction (Key feature!).
        $$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$
    4.  **Output Gate**: Determine next hidden state.
        $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
        $$ h_t = o_t \odot \tanh(C_t) $$

*   **Gradient Highway (Constant Error Carousel)**:
    Looking at the gradient of the cell state:
    $$ \frac{\partial C_t}{\partial C_{t-1}} = f_t + \dots $$
    If $f_t \approx 1$ (remember), the gradient passes through as identity, preserving error signals over long distances.

### GRU (Gated Recurrent Unit)
A simplified variant of LSTM that merges cell and hidden states.
*   **Equations**:
    $$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad (\text{Update Gate}) $$
    $$ r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad (\text{Reset Gate}) $$
    $$ \tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) $$
    $$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$
*   **Pros**: Fewer parameters than LSTM, often comparable performance.

### Seq2Seq (Sequence-to-Sequence) / Encoder-Decoder
Framework for mapping sequence $X$ to sequence $Y$ (e.g., Translation).

*   **Encoder**: Processes input sequence $x_1, \dots, x_N$ into a fixed-length **Context Vector** $v$ (usually the final hidden state $h_N$).
*   **Decoder**: Generates output sequence $y_1, \dots, y_M$ from $v$.
    $$ h^{dec}_t = \text{RNN}(h^{dec}_{t-1}, y_{t-1}, v) $$
*   **Teacher Forcing**:
    During training, the *ground truth* token $y_{t-1}^*$ is fed as input to step $t$, rather than the model's own prediction $\hat{y}_{t-1}$. This stabilizes training but can lead to "Exposure Bias" (model relies on perfect past during training but fails with its own errors during inference).
*   **Bottleneck Problem**: Compressing the entire source sentence info into a single vector $v$ limits performance for long sentences. This motivated the **Attention Mechanism**.

---

## 8. Attention & Transformer

### Scaled Dot-Product Attention
The core mechanism that maps a query and a set of key-value pairs to an output.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
*   **Inputs**: $Q$ (Query), $K$ (Key), $V$ (Value) are matrices where rows correspond to sequence tokens.
*   **Scaling Factor $\frac{1}{\sqrt{d_k}}$**: Prevents the dot products $QK^T$ from growing too large in magnitude. Large values push the Softmax function into regions with extremely small gradients (vanishing gradients), scaling helps maintain variance around 1.

### Multi-Head Attention (MHA)
Instead of a single attention function, we project queries, keys, and values $h$ times with different, learned linear projections to $d_k, d_k, d_v$ dimensions.
$$ \text{Head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{Head}_1, \dots, \text{Head}_h)W^O $$
*   **Benefit**: Allows the model to jointly attend to information from different representation subspaces at different positions.

### Transformer Block Architecture
A Transformer is composed of a stack of identical layers. Each layer has two sub-layers:
1.  **Multi-Head Self-Attention**:
    $$ y_1 = \text{LayerNorm}(x + \text{MultiHead}(x, x, x)) $$
2.  **Position-wise Feed-Forward Network (FFN)**:
    Applied to each position separately and identically. Usually two linear transformations with a ReLU/GELU activation in between.
    $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
    $$ y_2 = \text{LayerNorm}(y_1 + \text{FFN}(y_1)) $$
*   **Residual Connections & LayerNorm**: Crucial for training deep transformers. $\text{LayerNorm}(x + \text{Sublayer}(x))$.
*   **Positional Encoding**: Injected at the input (via addition) to provide order information, since Attention is permutation invariant.

### FlashAttention (IO-Aware Exact Attention)
Standard Attention implementation requires $O(N^2)$ memory to store the attention matrix $A = (QK^T)$, which often exceeds GPU high-bandwidth memory (HBM).
*   **Problem**: Materializing the $N \times N$ matrix $S = QK^T$ and $P = \text{softmax}(S)$ in HBM is slow due to memory bandwidth constraints.
*   **FlashAttention (Dao et al., 2022)**:
    *   **Tiling**: Computes attention in blocks (tiles) using on-chip SRAM (fast cache), never writing the full $N \times N$ matrix to HBM.
    *   **Recomputation**: Does not store the attention matrix for the backward pass; instead, it recomputes it on-the-fly.
    *   **Result**: Exact same output as standard attention, providing significant speedup (2-4x) and linear memory complexity with respect to sequence length in HBM, enabling much longer context windows.

---

## 9. GNN (Graph Neural Networks)

### Definition & Framework (MPNN)
Operates on data represented as graphs $G=(V, E)$. The core mechanism is **Message Passing**.
$$ h_v^{(k)} = \text{UPDATE}^{(k)} \left( h_v^{(k-1)}, \text{AGGREGATE}^{(k)}\left( \{ h_u^{(k-1)} \mid u \in \mathcal{N}(v) \} \right) \right) $$
*   **Permutation Equivariance**: The output $f(PAP^T, PX) = P f(A, X)$ ensures predictions heavily depend on graph structure, not node ordering.
*   **Inductive Bias**: Relational inductive bias, assuming dependencies exist defined by edges.

### Common Architectures

1.  **GCN (Graph Convolutional Network)** (Kipf & Welling, 2017)
    *   **Mechanism**: Approximates spectral graph convolutions (Chebyshev polynomials) to a simple first-order neighborhood average.
    *   **Update Rule**:
        $$ H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}) $$
        where $\tilde{A} = A + I$ (self-loops) and $\tilde{D}$ is the degree matrix of $\tilde{A}$.
    *   **Insight**: Effectively performs a weighted average of neighbor features (smoothing), acting as a low-pass filter on graph signals.

2.  **GAT (Graph Attention Network)** (Veličković et al., 2018)
    *   **Mechanism**: Assigns different importance weights $\alpha_{ij}$ to different neighbors using Self-Attention.
    *   **Update Rule**:
        $$ h_v' = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u \right), \quad \alpha_{vu} = \text{softmax}(\text{LeakyReLU}(a^T [Wh_v \| Wh_u])) $$
    *   **Insight**: Allows the model to focus on relevant neighbors, improving expressivity over fixed GCN weights. Supports heterogeneous graphs implicitly.

3.  **GIN (Graph Isomorphism Network)** (Xu et al., 2019)
    *   **Mechanism**: Uses **Sum** aggregation instead of Mean/Max to preserve structural information.
    *   **Update Rule**:
        $$ h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right) $$
    *   **Theory**: Proven to be as powerful as the **Weisfeiler-Lehman (1-WL) test** for graph isomorphism, whereas GCN/GraphSAGE are strictly less powerful.

### Challenges
*   **Oversmoothing**: In Deep GCNs, repeated averaging causes node representations to converge to the same value (indistinguishable), limiting depth.
*   **Over-squashing**: Information from distant nodes must be compressed into fixed-size vectors through a "bottleneck", losing information (critical in graphs with exponential expansion like trees).

### Inference Examples
1.  **Node Classification**: Classifying documents in a citation network or fraud detection in transaction graphs.
    *   *Input*: Features of all nodes + Partial labels. *Output*: Labels for unlabeled nodes.
2.  **Graph Classification**: Predicting molecular properties (toxicity, solubility).
    *   *Mechanism*: Use a **Readout Function** (Global Pooling) like Sum or Mean over all node embeddings to get a graph vector $h_G$.
3.  **Link Prediction**: Recommender systems (User-Item Bipartite Graph).
    *   *Mechanism*: Predict if an edge exists between user $u$ and item $i$ by computing score $s = h_u^T h_i$.
4.  **Heterogeneous/Multi-Graph Inference**: Knowledge Graphs with multiple edge types.
    *   *Mechanism*: Use Relational GCN (R-GCN) where weights $W_r$ are specific to edge type $r$.

---

## 10. Optimal Transport (OT)

### Concept
Provides a geometrically meaningful distance defining the "cost" to transform one probability distribution into another.

### Theoretical Definitions
*   **Wasserstein Distance (Earth Mover's Distance)**:
    $$ W(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|] $$
    Provides a weak topology where distributions with disjoint supports (common in high dimensions) still have meaningful distances and gradients, unlike Kullback-Leibler.
*   **Kantorovich-Rubinstein Duality**:
    $$ W_1(P, Q) = \sup_{\|f\|_L \le 1} \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{y \sim Q}[f(y)] $$

---

## 11. KAN (Kolmogorov-Arnold Networks)

### Theoretical Foundation: Kolmogorov-Arnold Representation Theorem
The theorem (1957) states that any multivariate continuous function $f(x_1, \dots, x_n)$ on a bounded domain can be represented as:
$$ f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^n \psi_{p,q}(x_p) \right) $$
where $\Phi_q$ and $\psi_{p,q}$ are continuous single-variable functions.

### KAN Architecture (Liu et al., 2024)
Inspired by this theorem, KANs differ fundamentally from MLPs.
*   **MLP**: Activate($\sum$ Weight $\times$ Input). (Fixed Activation, Learnable Weights).
*   **KAN**: $\sum$ LearnableFunction(Input). (Learnable Activation, Fixed Weights/Sum).

### Key Differences & Mechanism
1.  **Edges as Functions**: In KANs, the "activation functions" are placed on the **edges** (arrows) of the graph, not the nodes.
    $$ x^{(l+1)}_j = \sum_{i=1}^{N_l} \phi_{l, j, i}(x^{(l)}_i) $$
    where $\phi$ is a learnable 1D function parametrized by **B-splines**.
2.  **No Linear Weights**: There are no traditional weight matrices $W$. The "weights" are the coefficients of the spline basis functions.
3.  **Interpretability**: KANs can often be symbolically pruned to reveal the exact mathematical formula governing the data (e.g., discovering $f(x,y) = \sin(x) + y^2$).
4.  **Scaling Laws**: KANs have shown faster scaling properties (Accuracy vs Parameters) than MLPs for certain scientific / PDE solving tasks.


---


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


---


# Option Theory Cheatsheet

A comprehensive guide to analytical solutions for option pricing, stochastic calculus foundations, and the Greeks.

## 1. Fundamentals

### Stochastic Differential Equation (SDE)
A general 1-dimensional SDE is described as:

$$
dX_t = \mu(t, X_t) dt + \sigma(t, X_t) dW_t
$$

Where:
*   $X_t$: Stochastic process
*   $\mu$: Drift term
*   $\sigma$: Diffusion term
*   $W_t$: Standard Brownian Motion (Wiener Process)

### Ito's Lemma
For a stochastic process $X_t$ following the above SDE, the differential of a function $f(t, X_t)$ is:

$$
df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t
$$

Special case $X_t = W_t$ ($\mu=0, \sigma=1$):
$$
df(t, W_t) = \left( \frac{\partial f}{\partial t} + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} \right) dt + \frac{\partial f}{\partial x} dW_t
$$

### Geometric Brownian Motion (GBM)
Standard model for stock prices $S_t$:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

Applying Ito's Lemma to $f(S_t) = \ln S_t$:

$$
S_T = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)(T-t) + \sigma (W_T - W_t) \right)
$$

Properties:
*   $E[S_T | S_t] = S_t e^{\mu(T-t)}$
*   $Var[\ln(S_T/S_t)] = \sigma^2(T-t)$

## 2. Pricing Framework

### Risk-Neutral Measure ($\mathbb{Q}$)
In an arbitrage-free market, there exists a measure $\mathbb{Q}$ under which the discounted asset price is a martingale.
Stock price dynamics under $\mathbb{Q}$ (with continuous dividend yield $q$):

$$
dS_t = (r - q) S_t dt + \sigma S_t dW_t^\mathbb{Q}
$$

### Risk-Neutral Valuation
The value $V_t$ of a derivative with payoff $V_T$ is the discounted expected value under $\mathbb{Q}$:

$$
V_t = e^{-r(T-t)} E^\mathbb{Q} [ V_T | \mathcal{F}_t ]
$$

### Feynman-Kac Formula
The conditional expectation $V(t, x) = E^\mathbb{Q} [ e^{-r(T-t)} \psi(S_T) | S_t = x ]$ solves the Partial Differential Equation (PDE):

$$
\frac{\partial V}{\partial t} + (r-q) x \frac{\partial V}{\partial x} + \frac{1}{2} \sigma^2 x^2 \frac{\partial^2 V}{\partial x^2} - rV = 0
$$
Boundary Condition: $V(T, x) = \psi(x)$

## 3. Black-Scholes Model

### Black-Scholes PDE
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q) S \frac{\partial V}{\partial S} - rV = 0
$$

### Assumptions
*   Stock price follows GBM.
*   Risk-free rate $r$ and volatility $\sigma$ are constant.
*   No transaction costs or taxes.
*   Short selling is allowed.
*   Continuous trading is possible.

## 4. Analytical Solutions & Greeks

Notation:
*   $\tau = T - t$ (Time to maturity)
*   $N(x)$: Cumulative standard normal distribution function
*   $N'(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$: Standard normal probability density function

Standard Variables:
$$
d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}
$$
$$
d_2 = d_1 - \sigma\sqrt{\tau} = \frac{\ln(S/K) + (r - q - \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}
$$

### European Options (Vanilla)
*   **Call Payoff**: $\max(S_T - K, 0)$
*   **Put Payoff**: $\max(K - S_T, 0)$

### Digital Options (Binary)
These are options that pay a fixed amount depending on the position of the underlying price at maturity.

*   **Cash-or-Nothing Call**: Pays amount $Q$ (assume $Q=1$ below) if $S_T > K$.
*   **Cash-or-Nothing Put**: Pays amount $Q$ (assume $Q=1$ below) if $S_T < K$.
*   **Asset-or-Nothing Call**: Pays stock shares $S_T$ if $S_T > K$.
*   **Asset-or-Nothing Put**: Pays stock shares $S_T$ if $S_T < K$.

### Summary Table of Solutions and Greeks

| Measure | European Call | European Put | Digital Call (Cash-or-Nothing) | Digital Put (Cash-or-Nothing) |
| :--- | :--- | :--- | :--- | :--- |
| **Price** ($V$) | $S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2)$ | $K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1)$ | $e^{-r\tau} N(d_2)$ | $e^{-r\tau} N(-d_2)$ |
| **Delta** ($\Delta = \frac{\partial V}{\partial S}$) | $e^{-q\tau} N(d_1)$ | $-e^{-q\tau} N(-d_1)$ | $\frac{e^{-r\tau} N'(d_2)}{S\sigma\sqrt{\tau}}$ | $-\frac{e^{-r\tau} N'(d_2)}{S\sigma\sqrt{\tau}}$ |
| **Gamma** ($\Gamma = \frac{\partial^2 V}{\partial S^2}$) | $\frac{e^{-q\tau} N'(d_1)}{S \sigma \sqrt{\tau}}$ | $\frac{e^{-q\tau} N'(d_1)}{S \sigma \sqrt{\tau}}$ | $-\frac{e^{-r\tau} d_1 N'(d_2)}{S^2 \sigma^2 \tau}$ | $\frac{e^{-r\tau} d_1 N'(d_2)}{S^2 \sigma^2 \tau}$ |
| **Vega** ($\mathcal{V} = \frac{\partial V}{\partial \sigma}$) | $S e^{-q\tau} \sqrt{\tau} N'(d_1)$ | $S e^{-q\tau} \sqrt{\tau} N'(d_1)$ | $-\frac{e^{-r\tau} d_1 N'(d_2)}{\sigma}$ | $\frac{e^{-r\tau} d_1 N'(d_2)}{\sigma}$ |
| **Theta** ($\Theta = \frac{\partial V}{\partial t}$) | $-\frac{S e^{-q\tau} N'(d_1)\sigma}{2\sqrt{\tau}} - r K e^{-r\tau} N(d_2) + q S e^{-q\tau} N(d_1)$ | $-\frac{S e^{-q\tau} N'(d_1)\sigma}{2\sqrt{\tau}} + r K e^{-r\tau} N(-d_2) - q S e^{-q\tau} N(-d_1)$ | see note* | see note* |
| **Rho** ($\rho = \frac{\partial V}{\partial r}$) | $K \tau e^{-r\tau} N(d_2)$ | $-K \tau e^{-r\tau} N(-d_2)$ | $-\tau e^{-r\tau} N(d_2) + \frac{\sqrt{\tau}}{\sigma} e^{-r\tau} N'(d_2)$ | $-\tau e^{-r\tau} N(-d_2) - \frac{\sqrt{\tau}}{\sigma} e^{-r\tau} N'(d_2)$ |

*Note on Digital Theta*:
$$
\Theta_{DigCall} = r e^{-r\tau} N(d_2) + \frac{e^{-r\tau} N'(d_2)}{\sigma \sqrt{\tau}}\left(\frac{\ln(S/K)}{\tau} - (r - q - \frac{\sigma^2}{2})\right)
$$
(Often simplified using relationships between Greeks).

### Asset-or-Nothing
Notice that a European Call is essentially:
$$
\text{Call} = (\text{Asset-or-Nothing Call}) - K \times (\text{Cash-or-Nothing Call})
$$
Thus:
*   Asset-or-Nothing Call Price: $S e^{-q\tau} N(d_1)$
*   Asset-or-Nothing Put Price: $S e^{-q\tau} N(-d_1)$

## 5. Exotic Options (Path Dependent)

### Barrier Options
Options that become active (Knock-in) or null (Knock-out) if the asset price $S_t$ crosses a barrier $H$.
Solutions provided here are for **Single Barrier Options** (Reiner-Rubinstein).

#### Parameters and Definitions
*   $\tau = T - t$
*   $\lambda = \frac{r - q + \frac{1}{2}\sigma^2}{\sigma^2}$
*   $y = \frac{\ln(H^2 / (SK))}{\sigma\sqrt{\tau}}$
*   $x_1 = \frac{\ln(S/K)}{\sigma\sqrt{\tau}} + (1 + \lambda)\sigma\sqrt{\tau}$
*   $x_2 = \frac{\ln(S/K)}{\sigma\sqrt{\tau}} + (1 - \lambda)\sigma\sqrt{\tau}$ (Same as standard $d_2$)
*   $y_1 = \frac{\ln(H^2 / (SK))}{\sigma\sqrt{\tau}} + (1 + \lambda)\sigma\sqrt{\tau}$
*   $y_2 = \frac{\ln(H^2 / (SK))}{\sigma\sqrt{\tau}} + (1 - \lambda)\sigma\sqrt{\tau}$

**Standard Terms ($A$ - $F$ common notation)**:
(Assuming payout is Vanilla Call/Put if valid, no separate rebate for simplicity)

*   $A = \phi S e^{-q\tau} N(\phi x_1) - \phi K e^{-r\tau} N(\phi x_2)$  (Vanilla Price, $\phi=1$ for Call, $-1$ for Put)
*   $B = \phi S e^{-q\tau} N(\phi x_1) - \phi K e^{-r\tau} N(\phi x_2)$  (Same as A but with barrier condition logic)
*   $C = \phi S e^{-q\tau} (H/S)^{2(\lambda+1)} N(\eta y_1) - \phi K e^{-r\tau} (H/S)^{2\lambda} N(\eta y_2)$
*   $D = \phi S e^{-q\tau} (H/S)^{2(\lambda+1)} N(\eta y_1) - \phi K e^{-r\tau} (H/S)^{2\lambda} N(\eta y_2)$

#### 1. Down Options ($S_0 > H$)
Payoffs depend on price falling to $H$.

| Option Type | Condition | Formula |
| :--- | :--- | :--- |
| **Down-and-In Call** | $K > H$ | $(H/S)^{2\lambda} (S^c(y) - S^c(x_1))$ (Often simplified: CD-like terms) |
| **Down-and-Out Call** | $S_0 > H$ | **Standard**: $A - C$ (where $A$=Vanilla Call, $C$=Reflection) <br> $C_{do} = S N(x_1) - K e^{-r\tau} N(x_2) - (H/S)^{2\lambda} [ S N(y_1) - K e^{-r\tau} N(y_2) ]$ <br> (Note: Validity strictly for $K > H$. If $K < H$, formula adjusts). |
| **Down-and-In Put** | Any | $P_{di} = S e^{-q\tau} (H/S)^{2\lambda+2} N(y_1) - K e^{-r\tau} (H/S)^{2\lambda} N(y_2)$ |
| **Down-and-Out Put** | $K > H$ | $A - B + C - D$ (Complex interactions if $K$ vs $H$ varies). <br> **Simplified**: $P_{do} = P_{vanilla} - P_{di}$ |
| **Down-and-In Call (Full)** | | $S (H/S)^{2\lambda} N(y_1) - K e^{-r\tau} (H/S)^{2\lambda - 2} N(y_2)$ |

*To be rigorous, here is the full set of 8 formulas relative to Vanilla Price ($V_{BS}$):*

Let $\eta = 1$ for In, $-1$ for Out. Use indicator functions for $K$ relative to $H$. The breakdown is complex so standard texts define terms $A, B, C, D$.

**Analytical Formula Summary (Haug representation)**:

**Call Options ($S > H$, Down)**
1.  **Down-and-In Call ($K > H$)**: $C_{di} = S e^{-q\tau} (H/S)^{2\lambda} N(y_1) - K e^{-r\tau} (H/S)^{2\lambda - 2} N(y_2)$
2.  **Down-and-Out Call ($K > H$)**: $C_{do} = C_{vanilla} - C_{di}$
3.  **Down-and-In Call ($K < H$)**: $C_{di} = C_{vanilla} - C_{do}$ (See Out formula below)
4.  **Down-and-Out Call ($K < H$)**: $C_{do} = S e^{-q\tau} N(x_1) - K e^{-r\tau} N(x_2) - S e^{-q\tau} (H/S)^{2\lambda} N(y_1) + K e^{-r\tau} (H/S)^{2\lambda - 2} N(y_2)$

**Put Options ($S > H$, Down)**
5.  **Down-and-In Put ($K > H$)**: $P_{di} = P_{vanilla} - P_{do}$
6.  **Down-and-Out Put ($K > H$)**: $P_{do} = S e^{-q\tau} N(-x_1) - K e^{-r\tau} N(-x_2) - S e^{-q\tau} (H/S)^{2\lambda} N(-y_1) + K e^{-r\tau} (H/S)^{2\lambda - 2} N(-y_2)$
7.  **Down-and-In Put ($K < H$)**: $P_{di} = P_{vanilla}$ (Knock-in is certain if ITM?) No, careful. Actually $P_{di} = A (\text{put terms})$.
    *   Correct Logic: If $K < H$ and we are Down-and-In, we must hit $H$ to activate. Since $S > H > K$, if we hit $H$, we are not consistently ITM deeper.
    *   Formula: $P_{di} = -S e^{-q\tau} (H/S)^{2\lambda} N(-y_1) + K e^{-r\tau} (H/S)^{2\lambda - 2} N(-y_2)$

**Up Options ($S < H$, Up)**
(Symmetric to Down options by swapping $S \leftrightarrow K$, etc., but specific formulas exist).
*   **Up-and-In Call**: $A (\text{reflection})$
*   **Up-and-Out Call**: Vanilla - In
*   **Up-and-In Put**: Reflexive to Down-In Call.
*   **Up-and-Out Put**: Vanilla - In.

### Lookback Options

Definitions:
*   $S_T$: Price at maturity
*   $M_T = \max_{0 \le t \le T} S_t$ (Maximum)
*   $m_T = \min_{0 \le t \le T} S_t$ (Minimum)
*   $b = r - q$ (Carry cost)

#### 1. Floating Strike Call
Payoff: $C_T = S_T - m_T$ (Buy at min, sell at spot)
$$
C = S e^{-q\tau} N(a_1) - S e^{-q\tau} \frac{\sigma^2}{2b} N(-a_1) - m_t e^{-r\tau} N(a_2) + S e^{-r\tau} \frac{\sigma^2}{2b} e^{Y} N(-a_3)
$$
Where:
*   $a_1 = \frac{\ln(S/m_t) + (b + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$
*   $a_2 = a_1 - \sigma\sqrt{\tau}$
*   $a_3 = \frac{\ln(S/m_t) + (-b + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$
*   $Y = \frac{2b}{\sigma^2} \ln(S/m_t)$

#### 2. Floating Strike Put
Payoff: $P_T = M_T - S_T$ (Sell at max, buy at spot)
$$
P = M_t e^{-r\tau} N(-b_2) - S e^{-q\tau} N(-b_1) + S e^{-r\tau} \frac{\sigma^2}{2b} [ -(M_t/S)^{2b/\sigma^2} N(b_3) + e^{b\tau} N(b_3 + \frac{2b}{\sigma}\sqrt{\tau}) ]
$$
(Formula form varies by recursive $M_t$ logic, standard form above simplifies for $t=0, M_0=S_0$).
Standard $t=0$ ($M_0=S_0$):
$$
P = S e^{-q\tau} [ \frac{\sigma^2}{2b} N(-x_1) - N(-x_2) ] + S e^{-r\tau} (1 + \frac{\sigma^2}{2b}) N(x_2 - \frac{2b}{\sigma}\sqrt{\tau})
$$

#### 3. Fixed Strike Call
Payoff: $C_T = \max(M_T - K, 0)$
$$
C = S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2) + S e^{-r\tau} \frac{\sigma^2}{2b} [ -(S/K)^{-2b/\sigma^2} N(d_1 - \frac{2b}{\sigma}\sqrt{\tau}) + e^{b\tau} N(d_1) ]
$$
(Note: $M_t$ term omitted for $t=0$ case where $M_0=S_0 < K$). If $M_t > K$, split into certainty part + option part.

#### 4. Fixed Strike Put
Payoff: $P_T = \max(K - m_T, 0)$
$$
P = K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1) + S e^{-r\tau} \frac{\sigma^2}{2b} [ (S/K)^{-2b/\sigma^2} N(-d_1 + \frac{2b}{\sigma}\sqrt{\tau}) - e^{b\tau} N(-d_1) ]
$$
(For $t=0$ case where $m_0=S_0 > K$).

### Asian Options (Geometric)

### Asian Options
Payoff depends on the average price $A_T$.

**Geometric Asian Option** (Closed Form):
Average defined as $G_T = \exp\left( \frac{1}{T} \int_0^T \ln S_t dt \right)$.
$G_T$ follows a lognormal distribution. We can use the Black-Scholes formula with adjusted parameters:
*   Volatility: $\sigma_{adj} = \sigma / \sqrt{3}$
*   Drift: $b_{adj} = \frac{1}{2}(r - q - \frac{1}{2}\sigma^2) + \frac{1}{6}\sigma^2$ (Example approximation for drift term depends on precise average definition)

**Arithmetic Asian Option**:
Approximations (e.g., Edgeworth expansion, Moment matching) or Monte Carlo are required. No exact closed form.

## 6. American Options

Exercisable at possible any time $t \le T$.

### Finite Horizon American Put
*   **No analytical closed-form solution**.
*   **Binomial Trees**: CRR model.
*   **Finite Difference**: Solve the PDE/Variational Inequality numerically.
*   **BBAW Approximation**: Splits value into European value + Early exercise premium.

### Perpetual American Put
Infinite maturity ($T \to \infty$).
The solution becomes time-independent (ODE).

$$
P(S) = \frac{K}{1 - \gamma} \left( \frac{(\gamma - 1)S}{\gamma K} \right)^\gamma
$$

Optimal Exercise Boundary:
$$
S^* = \frac{\gamma}{\gamma - 1} K
$$

where $\gamma$ is the negative root of the characteristic equation (assuming $q=0$ for simplicity):
$$
\frac{1}{2}\sigma^2 \gamma(\gamma - 1) + r \gamma - r = 0 \quad \Rightarrow \quad \gamma = \frac{-(r - \frac{\sigma^2}{2}) - \sqrt{(r - \frac{\sigma^2}{2})^2 + 2\sigma^2 r}}{\sigma^2}
$$

## 7. Delta Hedging & PnL Dynamics

### Theory: Market Completeness
In the Black-Scholes framework, risk can be completely eliminated because the market is **Complete**. This allows the creation of a **Replicating Portfolio** that perfectly mimics the option's behavior using the underlying asset $S$ and cash (bond).

### 1. PnL Decomposition
For a Delta-Hedged portfolio (Short Option + Long Stock), the Total PnL at maturity $T$ is:

$$
\text{Total PnL} = \underbrace{\text{Premium Received}}_{V_0} - \underbrace{\text{Option Payoff}}_{V_T} - \underbrace{\text{Total Rebalancing Cost}}_{\text{Hedge Cost}}
$$

If the hedge is perfect (and assumptions hold), $\text{Total PnL} \approx 0$ (ignoring interest rate for simplicity).

#### Rebalancing Cost (RC) Forms
Let $h_t = \Delta_t$ be the number of shares held. The cost can be analyzed in two ways using integration by parts ($d(hS) = h dS + S dh$):

**1. Cash Flow View ($\int S dh$)** - *Practical View*
Focuses on the cash spent/received to adjust the position size.
$$
RC = \int_0^T S_t dh_t \approx \sum_{i=1}^N S_i (h_i - h_{i-1})
$$
*   **Intuition**: If you are Short Gamma (Short Option), you buy stock when $S$ rises ($dh > 0$ at high $S$) and sell when $S$ falls ($dh < 0$ at low $S$). You consistently **Buy High, Sell Low**, incurring positive cost ($RC > 0$).

**2. Capital Gain View ($\int h dS$)** - *Theoretical View*
Focuses on the trading gains/losses from holding the stock.
$$
RC = h_T S_T - h_0 S_0 - \int_0^T h_t dS_t
$$
*   $\int h dS$: The cumulative profit from holding $h_t$ shares while price changes by $dS_t$.
*   Since Payoff ($V_T$) is replicated by Initial Wealth ($V_0$) + Trading Gains ($\int h dS$):
    $$
    V_T = V_0 + \int_0^T \Delta_t dS_t \quad (\text{if } r=0)
    $$
    This proves PnL is zero.

### 2. Hedging Simulation Table (Symbolic)
**Scenario**: Short 1 Call Option ($V$), Hedged with Stock ($S$).
**Objective**: Delta Neutral ($\Delta_{portfolio} = 0 \implies h = \Delta_{call}$).

| Time ($t$) | Stock ($S_t$) | Option ($V_t$) | Delta ($\Delta_t$) | Stock Pos ($h_t$) | Trade ($dh_t$) | Trade Cost ($S_t dh_t$) | Cash Balance ($B_t$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | $S_0$ | $V_0$ | $\Delta_0$ | $h_0 = \Delta_0$ | $+\Delta_0$ | $S_0 \Delta_0$ | $B_0 = V_0 - S_0 \Delta_0$ |
| **1** | $S_1$ | $V_1$ | $\Delta_1$ | $h_1 = \Delta_1$ | $\Delta_1 - \Delta_0$ | $S_1 (h_1 - h_0)$ | $B_1 = B_0 - S_1 dh_1$ |
| **2** | $S_2$ | $V_2$ | $\Delta_2$ | $h_2 = \Delta_2$ | $\Delta_2 - \Delta_1$ | $S_2 (h_2 - h_1)$ | $B_2 = B_1 - S_2 dh_2$ |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **k** | $S_k$ | $V_k$ | $\Delta_k$ | $h_k$ | $h_k - h_{k-1}$ | $S_k dh_k$ | $B_k = B_{k-1} - S_k dh_k$ |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **T** | $S_T$ | $V_T$ | $\Delta_T$ | $h_T$ | | | $B_T = B_{T-1} - S_T (h_T - h_{T-1})$ |

**Final Portfolio Value**:
$$
\Pi_T = \underbrace{B_T}_{\text{Cash}} + \underbrace{h_T S_T}_{\text{Stock liquidation}} - \underbrace{V_T}_{\text{Option Payoff}}
$$

If hedging is continuous and parameters match ($r=0$ case for simplicity):
$$
\Pi_T \approx 0
$$
This demonstrates that the initial premium $V_0$ plus trading gains/losses exactly covers the final payoff $V_T$.

### 3. Impact of Volatility
The PnL of a delta-hedged strategy depends on the difference between Realized Volatility ($\sigma_{real}$) and Implied Volatility ($\sigma_{imp}$).

$$
\text{PnL} \approx \frac{1}{2} S^2 \Gamma (\sigma_{imp}^2 - \sigma_{real}^2) dt
$$

*   **Short Gamma** (Short Option): You lose if market is more volatile than expected ($\sigma_{real} > \sigma_{imp}$). "Buy High Sell Low" costs overwhelm the Premium.
*   **Long Gamma** (Long Option): You gain if market is more volatile than expected. You "Buy Low Sell High" during rebalancing.

---
**References**
*   John C. Hull, "Options, Futures, and Other Derivatives"
*   Steven E. Shreve, "Stochastic Calculus for Finance II"


---


# Polars Cheatsheet

A comprehensive guide to [Polars](https://pola.rs/), a localized (single-machine) high-performance DataFrame library for Python, built on Rust and Apache Arrow.

## 🐍 Pandas vs. 🐻‍❄️ Polars: Key Differences & Syntax Mapping

| Feature | Pandas (`pd`) | Polars (`pl`) | Key Difference |
| :--- | :--- | :--- | :--- |
| **Index** | Has a row Index (often implicit). | **No Index.** Row numbers are implicit. | Polars doesn't use indexes for lookups; it uses physical memory offsets. |
| **Missing Values** | `NaN` (float), `None` (object), `pd.NA`. | **`null`** (applies to all types). | Consistent missing value handling across all data types. |
| **Selection** | `df[['a', 'b']]` | `df.select(['a', 'b'])` | Polars discourages `[]` for col selection in favor of `.select()`. |
| **Filtering** | `df[df['a'] > 5]` | `df.filter(pl.col('a') > 5)` | Uses `.filter()` with expressions. |
| **New Column** | `df['sum'] = df['a'] + df['b']` | `df.with_columns((pl.col('a') + pl.col('b')).alias('sum'))` | **Immutable.** `.with_columns()` returns a new DataFrame. |
| **Grouping** | `df.groupby('a').agg({'b': 'sum'})` | `df.group_by('a').agg(pl.col('b').sum())` | `group_by` (v1.0+) vs `groupby`. Aggregation uses expressions. |
| **Sorting** | `df.sort_values('a')` | `df.sort('a')` | Simpler method name. |
| **Strings** | `df['s'].str.upper()` | `df.select(pl.col('s').str.to_uppercase())` | String methods under `.str` namespace in expressions. |
| **Unpivoting** | `df.melt(...)` | `df.unpivot(...)` | `unpivot` is preferred over `melt` in recent Polars versions. |
| **Execution** | Eager (mostly). | **Lazy & Eager.** | Polars emphasizes `LazyFrame` for query optimization. |

### 💡 Key Philosophy
1.  **Expressions are King**: Nearly everything in Polars (selection, filtering, aggregation, column creation) uses **Expressions** (`pl.col(...)`). They are lazy, composable, and optimizer-friendly.
2.  **No Index**: Use standard columns for identifiers. Valid index usage in Pandas (like `loc`) is often handled by `filter` in Polars.
3.  **Parallelization**: Polars automatically parallelizes operations utilizing all available cores without custom logic.
4.  **Method Chaining**: Writing almost everything as a chain of methods allows the Polars query optimizer to see the "whole picture," creating a more efficient execution plan than step-by-step Pandas execution.

---

## 🚀 Installation & Basics

```bash
pip install polars
# For improved performance on old CPUs or specific hardware, check documentation for optimized builds (e.g. polars-lts-cpu)
```

```python
import polars as pl
import pandas as pd

# From Dictionary
data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df = pl.DataFrame(data)

# From Pandas
df_pd = pd.DataFrame(data)
df = pl.from_pandas(df_pd)
```

---

## 💾 Input / Output

| Format | Read | Write (Eager) | Write (Lazy) |
| :--- | :--- | :--- | :--- |
| **CSV** | `pl.read_csv("file.csv")` | `df.write_csv("file.csv")` | `df_lazy.sink_csv(...)` |
| **Parquet** | `pl.read_parquet("file.parquet")` | `df.write_parquet("file.parquet")` | `df_lazy.sink_parquet(...)` |
| **JSON** | `pl.read_json("file.json")` | `df.write_json("file.json")` | `df_lazy.sink_json(...)` |
| **Database** | `pl.read_database_uri(query, uri)` | `df.write_database(...)` | - |

**Lazy Scanning** (Does not load data into memory immediately):
```python
lf = pl.scan_csv("data.csv")
lf = pl.scan_parquet("data.parquet")
```

---

## 🔍 Selection & Viewing

```python
df.head(5)        # First 5 rows
df.tail(5)        # Last 5 rows
df.glimpse()      # Dense overview of data types and values (like R's glimpse)
df.schema         # Dict of column names to DataTypes
df.columns        # List of column names

# Select specific columns
df.select("a", "b")
df.select(["a", "b"])
df.select(pl.col("a"), pl.col("b"))

# Select with exclusion
df.select(pl.all().exclude("b"))

# Select by type
# Select by type using Selectors (cs)
import polars.selectors as cs
df.select(cs.numeric())          # All numeric columns
df.select(cs.string())           # All string columns
df.select(cs.by_name("res.*"))   # Columns matching regex
df.select(~cs.numeric())         # detailed negation (NOT numeric)
```

---

## 🔽 Filtering (Rows)

Polars uses `.filter()` with boolean expressions.

```python
# Simple filter
df.filter(pl.col("a") > 2)

# Multiple conditions (& for AND, | for OR)
df.filter((pl.col("a") > 1) & (pl.col("b") < 10))

# Is In / Is Null
df.filter(pl.col("a").is_in([1, 3, 5]))
df.filter(pl.col("a").is_null())
df.filter(pl.col("a").is_not_null())
```

---

## ⚡ Expressions: The Power House

Expressions (`pl.col(...)`) define transformations solely on columns. They can be used in `select`, `with_columns`, `filter`, `group_by`, etc.

### Core Arithmetic & Logic
```python
pl.col("a") + pl.col("b")
pl.col("a") * 2
(pl.col("a") > 0).alias("is_positive") # Rename result
pl.when(pl.col("a") > 5).then("High").otherwise("Low") # If-Else style
```

### String Operations (`.str`)
```python
pl.col("s").str.to_uppercase()
pl.col("s").str.contains("pattern")
pl.col("s").str.replace("b", "B")
pl.col("s").str.slice(0, 3) # Substring
pl.col("s").str.split(" ")
```

### Date & Time (`.dt`)
```python
pl.col("date").dt.year()
pl.col("date").dt.month()
pl.col("date").dt.strftime("%Y-%m-%d")
pl.col("date").dt.add_business_days(5)
```

### List / Array Operations (`.list`)
```python
pl.col("list_col").list.len()
pl.col("list_col").list.get(0)      # First element
pl.col("list_col").list.join(", ")  # Join elements to string
pl.col("list_col").list.explode()   # Unnest list to rows
```

---

## 🛠 Column Manipulation

**`with_columns`** is the primary method to add, modify, or compute columns. It runs operations in parallel.

```python
# Add/Update multiple columns
df = df.with_columns(
    (pl.col("a") * 2).alias("a_doubled"),
    (pl.col("b") / 10).alias("b_scaled"),
    pl.lit("constant_value").alias("const") # Literal value
)

# Casting types
df = df.with_columns(
    pl.col("a").cast(pl.Int32),
    pl.col("date_str").str.to_datetime("%Y-%m-%d")
)

# Sorting / Renaming / Dropping
df.sort("a", descending=True)
df.rename({"old_name": "new_name"})
df.drop("col_to_remove")
```

---

## 📊 Aggregation & Grouping

Use `group_by` (preferred over `groupby`) followed by `agg`.

```python
df.group_by("category").agg(
    pl.len().alias("count"),            # Count rows in group
    pl.col("val").sum().alias("sum"),   # Sum of 'val'
    pl.col("val").mean(),               # Mean of 'val'
    pl.col("val").max(),                # Max of 'val'
    pl.col("val").first(),              # First value
    pl.col("names").list()              # Collect values into a list
)
```

### Window Functions (`.over()`)
Compute aggregations *without* collapsing groups (like SQL window functions). Extremely powerful.

```python
# Add a column with group average alongside original data
df.with_columns(
    pl.col("val").mean().over("category").alias("cat_mean")
)

# Calculate difference from group mean
df.with_columns(
    (pl.col("val") - pl.col("val").mean().over("category")).alias("diff_from_mean")
)
```

---

## ⚡ Performance Tips: The Power of Method Chaining

In Pandas, you often assign intermediate results to variables. In Polars, **chaining methods** is critical for performance, especially in Lazy mode. It allows the Query Optimizer to:
1.  **Predicate Pushdown**: Apply filters *before* loading data or expensive operations.
2.  **Projection Pushdown**: Load only necessary columns.
3.  **Common Subexpression Elimination**: Avoid redundant calculations.

### Pandas Style (Avoid)
```python
# ❌ Intermediate steps block optimization and memory release
df1 = pl.read_csv("data.csv")
df2 = df1.filter(pl.col("status") == "active")
df3 = df2.select(["id", "val"])
result = df3.group_by("id").agg(pl.col("val").sum())
```

### Polars Style (Preferred)
```python
# ✅ Chaining allows Polars to optimize the entire query
result = (
    pl.scan_csv("data.csv")  # Lazy load
    .filter(pl.col("status") == "active")
    .select("id", "val")
    .group_by("id")
    .agg(pl.col("val").sum())
    .collect()
)
```

---

## 🔗 Joins & Combination

### Joins
```python
# Standard SQL-style joins: inner, left, outer, semi, anti, cross
df_a.join(df_b, on="id", how="inner")
df_a.join(df_b, left_on="a_id", right_on="b_id", how="left")
```

### Concatenation
```python
# Vertical (Stacking rows)
pl.concat([df1, df2], how="vertical") # Equivalent to vstack

# Horizontal (Stacking columns - MUST have same height)
pl.concat([df1, df2], how="horizontal") # Equivalent to hstack
```

---

## 🔄 Reshaping

| Operation | Method | Description |
| :--- | :--- | :--- |
| **Pivot** | `.pivot()` | Long -> Wide. |
| **Unpivot** | `.unpivot()` | Wide -> Long. (Formerly `melt`) |
| **Transpose** | `.transpose()` | Swap rows and columns. |

```python
# Pivot
df.pivot(on="year", index="country", values="gdp")

# Unpivot (Melt)
df.unpivot(index=["country"], on=["2020", "2021"], variable_name="year", value_name="gdp")
```

---

## 🚫 Missing Data Handling

```python
# Fill Nulls
df.with_columns(pl.col("a").fill_null(0))
df.with_columns(pl.col("a").fill_null(strategy="forward"))

# Drop Nulls
df.drop_nulls()               # Drop row if ANY column is null
df.drop_nulls(subset=["a"])   # Drop row if 'a' is null

# Replace NaNs (Floats only, distinct from Null)
df.with_columns(pl.col("val").fill_nan(0))
```

---

## 💤 Lazy API (Performance Booster)

The "Lazy" API records operations without executing them. When `collect()` is called, Polars builds a query plan, optimizes it (predicate pushdown, projection pushdown), and executes it.

```python
# 1. Start with a LazyFrame
lf = pl.scan_csv("huge_file.csv")

# 2. Chain operations (nothing happens yet)
q = (
    lf.filter(pl.col("val") > 100)
    .group_by("group")
    .agg(pl.col("val").mean())
)

# 3. Optimize & Execute
df_result = q.collect()  # Returns eager DataFrame

# Or Stream results (for larger-than-RAM datasets)
q.sink_parquet("output.parquet")
```

### Inspecting Plans
```python
q.explain() # Print the unoptimized query plan
q.explain(optimized=True) # Print the optimized plan
```

---

## ⚙️ Configuration

```python
# Adjust display settings
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(10)
pl.Config.set_fmt_str_lengths(50)
```


---


# Productivity & Engineering Philosophy

A comprehensive guide to "Intellectual Production", "Engineering Mindset", and "Professional Conduct".

> "Efficiency is doing things right; effectiveness is doing the right things." - Peter Drucker

## 1. Deep Work (Focus Management)
> Source: *Deep Work (Cal Newport)*

### The Core Hypothesis
The ability to perform deep work is becoming increasingly rare at exactly the same time it is becoming increasingly valuable in our economy. As a consequence, the few who cultivate this skill, and then make it the core of their working life, will thrive.

### Strategies for Focus
*   **The 4 Disciplines of Execution (4DX)**:
    1.  **Focus on the Wildly Important**: Don't try to do everything. Pick 1-2 ambitious goals.
    2.  **Act on Lead Measures**: Track metrics you can control (e.g., "Hours of Deep Work") rather than lag measures (e.g., "features shipped").
    3.  **Keep a Scoreboard**: Visualize your deep work hours.
    4.  **Create a Cadence of Accountability**: Weekly reviews of your focus performance.
*   **Roosevelt Dashes**:
    *   Brief but extremely intense periods of work. Estimate how long a task *should* take, then drastically cut that time to force extreme concentration.
*   **The Shutdown Ritual**:
    *   End your workday with a strict ritual. Review tasks, plan the next day, and say a "termination phrase" (e.g., "Shutdown Complete").
    *   **Why?**: Incomplete tasks linger in your mind (Zeigarnik Effect). The ritual frees your brain to rest.

### Scheduling Philosophies
*   **Monastic**: No distractions, ever (unrealistic for most).
*   **Bimodal**: Long periods (days/weeks) of isolation, followed by normal life.
*   **Rhythmic**: Habitual blocks (e.g., 6:00 AM - 9:00 AM every day). Best for office workers.
*   **Journalistic**: Seizing 30-minute gaps whenever possible. Requires high mental agility.

## 2. Managing Complexity (Cognitive Load)
> Source: *A Philosophy of Software Design (John Ousterhout)*

### The Root of All Evil
Complexity is the accumulation of dependencies and obscurities. It increases the cognitive load required to make even simple changes.

### Tactical vs. Strategic Programming
*   **Tactical**: "Just get it working." Shortcuts, tech debt, spaghetti code. Fast now, slow forever after.
*   **Strategic**: "Investing in design." It takes 10-20% longer initially but keeps development velocity high over time.
    > *Rule*: If you have to choose, always choose the strategic approach unless your startup dies tomorrow without the feature.

### Design Principles
*   **Deep Modules**:
    *   Expose a *simple* interface for *complex* functionality.
    *   **Information Hiding**: The implementation details that are most likely to change should be completely hidden.
*   **Comments as "Why"**:
    *   Code explains *what* is happening. Comments explain *why* it is happening (design decisions, non-obvious constraints).
    *   Write comments *before* code to clarify your design thinking.
*   **Define Errors Out of Existence**:
    *   Instead of throwing exceptions, redefine the semantics so the exception is impossible.
    *   *Example*: A file delete method dealing with a non-existent file. Instead of throwing `FileNotFound`, simple return `Success` (the desired state "file is gone" is achieved).

## 3. Teamwork & Culture (HRT Principle)
> Source: *Team Geek (Brian W. Fitzpatrick)*

### The HRT Triad
*   **Humility**: You are not the center of the universe. You make mistakes. Be open to self-correction.
*   **Respect**: Treat colleagues as competent professionals. Assume positive intent.
*   **Trust**: Believe that others can do the job. Delegate authority, not just tasks.

### The Bus Factor
*   The number of key people who, if hit by a bus (or quit), would doom the project.
*   **Goal**: Increase the bus factor. Share knowledge, document everything, avoid "hero programmers".

### Communication Protocols
*   **Mission Statements**: Define a clear direction. "We are building X to solve Y for Z."
*   **Consensus vs. Voting**: Avoid voting. It creates winners and losers. Strive for consensus, but if deadlocked, the tech lead decides.
*   **The "No Asshole" Rule**: brilliance is not an excuse for toxic behavior. Toxic high-performers destroy team net productivity.

## 4. Problem Solving (McKinsey Style)
> Source: *Bulletproof Problem Solving (Conn & McLean)*

### The Seven Steps Overview
1.  **Define**: Context, criteria for success, constraints, stakeholders.
2.  **Disaggregate**: Break it down.
3.  **Prioritize**: Pareto principle (80/20).
4.  **Workplan**: Who, what, when.
5.  **Analyze**: Heuristics -> Deep Analysis.
6.  **Synthesize**: Findings -> Insights.
7.  **Communicate**: Drive action.

### Disaggregation Techniques
*   **Logic Trees**:
    *   **Component Tree**: "What is it?" (Decompose a system).
    *   **Hypothesis Tree**: "Why might this be?" (Testable hypotheses).
*   **MECE**: Mutually Exclusive, Collectively Exhaustive. No overlaps, no gaps.

### The "One Day Answer"
*   Before spending weeks on research, formulate your "best guess" answer on Day 1.
*   This hypothesis drives your analysis. You are trying to prove or disprove it, not just "looking at data".

## 5. Technical Writing & Communication
> Source: *The Pyramid Principle (Barbara Minto) / Technical Writing Guidelines*

### The SCQA Framework (Storytelling)
To get attention, frame your document/presentation as a story:
1.  **S**ituation: The undeniable, non-controversial context. "We use Python 3.8."
2.  **C**omplication: The problem that disrupts the situation. "Security support for 3.8 ends next month."
3.  **Q**uestion: The natural question arises. "What should we do?"
4.  **A**nswer: Your solution (The BLUF). "Migrate to Python 3.12 immediately."

### Micro-Writing Tips
*   **Active Voice**: "The server received the request" (Better) vs "The request was received by the server".
*   **Strong Verbs**: Avoid "make", "do", "get". Use "generate", "calculate", "retrieve".
*   **Lists**: If you have 3+ items, make a bulleted list.

---

## 6. Professional Conduct (Communicating like a Pro)

How to behave to be trusted by Seniors, Managers, Juniors, and Clients.

### Managing Up (Bosses / Managers)
*   **No Surprises**:
    *   Never let your boss be surprised by bad news from someone else.
    *   Report bad news *immediately*, but always pair it with a potential solution or a mitigation plan.
    *   *Bad*: "The DB crashed."
    *   *Good*: "The DB crashed. We've failed over to the replica. I'm investigating the root cause and will update in 30 mins."
*   **Solution-Oriented**:
    *   Don't just bring problems. Bring proposals.
    *   "We have a problem X. I recommend we do Y. Option Z is also possible but riskier."
*   **Manage Expectations**:
    *   Under-promise, over-deliver. If a task takes 3 days, say 4-5.
    *   If a deadline is at risk, communicate it *days* in advance, not hour of.

### Interacting with Seniors / Mentors
*   **Respect Their Time (The "15 Minute Rule")**:
    *   Before asking, spend 15 minutes trying to solve it yourself (Google, Docs, Debug).
    *   Don't spend *hours* spinning your wheels.
*   **Ask High-Quality Questions**:
    *   State: 1. What you want to do. 2. What you tried. 3. What exact error you got. 4. What you think might be the cause.
*   **Close the Loop**:
    *   If they give you advice, come back later and say "That worked, thanks!" or "It didn't work because..."

### Leading Juniors
*   **Psychological Safety**:
    *   Explicitly tell them: "It's okay to make mistakes. It's okay to ask 'stupid' questions."
*   **Delegate Context, Not Just Tasks**:
    *   Don't say "Write this function."
    *   Say "We need to fix this user bug. I think this function is the place. Can you look into it?"
*   **Code Review as Mentorship**:
    *   Explain *why* you are requesting a change. Link to documentation.

### Client / Stakeholder Management
*   **Speak Their Language**:
    *   They don't care about "Refactoring the React hook".
    *   They care about "Making the checkout page load 50% faster to increase conversion."
*   **Reliability is King**:
    *   It is better to be consistently average speed than wildly unpredictable.
    *   If you say "Tuesday", it must be Tuesday.
*   **The "Yes, and..." or "No, but..."**:
    *   Avoid hard "No".
    *   "We can't do that feature by Friday (No), *but* we can ship the core version and follow up next week (Alternative)."


---


# The Seven Sins of Quantitative Investing

> **Based on Deutsche Bank Quantitative Strategy Research**

This cheatsheet outlines the seven most common biases and mistakes ("Sins") that occur during the backtesting of quantitative investment models, along with their remedies.

---

## 1. Survivorship Bias

**The Sin:**
Considering only companies that are currently in existence, ignoring those that have gone bankrupt, merged, or been delisted. This artificially inflates historical performance because "dead" companies usually performed poorly.

**Examples:**
- **Russell 3000:** "Survivor-only" universe yields significantly higher returns.
- **Credit Risk Models:** Testing on survivors makes risky firms appear to have highest returns (7x better) because failed risky firms are excluded.

**The Remedy:**
- **Use Point-in-Time (PIT) Data:** Include "dead" companies active during the historical period.
- **Handle Missing Data:** Treat companies as part of the universe until they formally exit; don't filter out based on missing trailing data if it means excluding delistings.

---

## 2. Look-ahead Bias

**The Sin:**
Using information in a backtest that was not actually available to investors at that time.

**Examples:**
- **Earnings Data Lag:** Assuming data is available on quarter-end, when it's actually reported 1-3 months later.
- **Restated Data:** Using revised historical data instead of preliminary data originally released.
- **Split Adjustments:** Using split-adjusted prices for price-level strategies (e.g., "Buy stocks under $5").

**The Remedy:**
- **Lag Data:** Mechanically lag financial data (e.g., 3 months) if PIT timestamps are unavailable.
- **Use Unadjusted Prices:** For strategies based on absolute price levels.
- **Strict PIT Databases:** Use databases with exact public availability timestamps.

---

## 3. The Sin of Storytelling

**The Sin:**
Post-hoc rationalization. Creating a plausible "story" to explain a statistical anomaly after seeing the results.

**Examples:**
- **Leverage:** Interpreting performance as "risk premium" vs "penalizing distress" depending on the outcome.
- **Tech Bubble (1998-2000):** Inventing stories that "traditional valuation is dead" when Value failed, only for it to rebound.

**The Remedy:**
- **Long-Term Validation:** Test over long horizons (20+ years) across multiple economic cycles.
- **Skepticism:** Be wary of theories created *after* data analysis.

---

## 4. Data Mining (and Snooping)

**The Sin:**
"Torturing the data until it confesses." Running hundreds of variations and picking the best one, leading to overfitting.

**Examples:**
- **72-Factor Experiment:** Picking best factors from a large pool yields high in-sample Sharpe (0.7) but zero out-of-sample return.

**The Remedy:**
- **Out-of-Sample Testing:** Always withhold data for testing (different time period or region).
- **Economic Intuition:** Select factors based on logic *before* testing.
- **Lock the Validation Set:** Do not touch validation data until the model is finalized.

---

## 5. Signal Decay and Turnover

**The Sin:**
Ignoring transaction costs and signal decay speed. High theoretical returns can turn into losses after costs.

**Examples:**
- **Short-Term Reversals:** "Buying yesterday's losers" has high returns but requires high turnover; costs eat the profit.
- **Implementation Delay:** Trading at "Open" instead of "Close" can degrade fast-moving signals.

**The Remedy:**
- **Net-of-Cost Testing:** Simulate returns after deducting realistic commissions and impact costs.
- **Staggered Rebalancing:** Rebalance a small portion (e.g., 3-4%) monthly instead of all at once to keep turnover manageable.

---

## 6. Outliers

**The Sin:**
Mishandling extreme data points. Including errors ruins results; excluding valid extremes removes risk info.

**Examples:**
- **1994 Earnings Distortion:** One company with 300,000% earnings yield distorted the index average.
- **Ranking Momentum:** Ranking (0-100) hurts Momentum strategies which rely on the *magnitude* of the price jump (~35% power loss).

**The Remedy:**
- **Winsorization:** Cap extreme values (e.g., 1st/99th percentile) instead of deleting.
- **Context-Aware Processing:** Use ranking for noisy data; preserve magnitude for distance-based factors (Momentum).

---

## 7. The Asymmetric Pattern and Shorting Cost

**The Sin:**
Assuming Shorting is as easy/cheap as Buying.

**Examples:**
- **Borrowing Costs:** "Hard-to-borrow" costs often eat up alpha for shorting bad companies.
- **Asymmetry:** Value works best Long; Momentum often works best Short.

**The Remedy:**
- **Explicit Constraints:** Assume higher costs for shorting; prohibit shorting "hard-to-borrow" stocks.
- **Diversification:** Increase holdings (e.g., to 500+) to reduce idiosyncratic risk if shorting is constrained.


---


# Research Hacks & Writing Cheatsheet

A collection of productivity hacks, writing guidelines, and inspiration for researchers.

## 🚀 Research Productivity Hacks

### Workflow & Focus
- **SMART Goals**: Set **S**pecific, **M**easurable, **A**ttainable, **R**elevant, **T**ime-bound goals for daily research.
- **The Pomodoro Technique**: Work in focused 25-minute intervals with 5-minute breaks to maintain high concentration.
- **Time Blocking**: Dedicate specific blocks of time for deep work (writing, analysis) vs. shallow work (emails, admin).
- **"Eat the Frog"**: Tackle your most difficult or important task first thing in the morning.
- **Digital Minimalism**: Turn off notifications and close irrelevant tabs during deep work sessions.

### Literature Search & Management
- **Connected Papers / Litmaps**: Use these tools to visualize citation networks and find relevant papers you might have missed.
- **Consensus**: AI search engine that extracts claims and evidence from papers to answer questions directly.
- **Mendeley / Zotero**: Essential for citation management. Use browser plugins to save papers instantly.
- **Systematic Filenaming**: Use a consistent format (e.g., `Year_Author_Topic.pdf`) to make searching your local library easier.

---

## ✍️ Writing Eye-Catching Papers

### The "Hook" for Abstracts & Introductions
Your Abstract and Introduction are the paper's "advertisement." Make them count.

#### Abstract (The "Elevator Pitch")
- **Be Concise**: 150–250 words maximum.
- **Structure**:
    1.  **Context**: One sentence on why the general topic matters.
    2.  **Gap**: One sentence on what is unknown or the specific problem.
    3.  **Action**: What did you do? (Methods in brief).
    4.  **Findings**: The most important numbers/results.
    5.  **Takeaway**: Why does this finding matter? (Implications).
- **Tip**: Write the Abstract *last*, but plan it *first*.

#### Introduction (The "Funnel")
- **Start Broad**: Begin with a "Hook"—a surprising statistic, a bold statement, or a clearly defined problem that affects many.
- **Narrow Down**: Transition from the general problem to your specific niche.
- **State the Gap**: Explicitly state what is missing in current literature ("However, previous studies have failed to address...").
- **The Solution**: Briefly state your hypothesis or objective.
- **The Map**: (Optional) In the last paragraph, outline how the paper is structured.

---

## 📄 Essential Components by Section

### 1. Abstract
- **Objective**: What is the problem?
- **Methods**: How did you solve it?
- **Results**: What did you find? (Key metrics).
- **Conclusion**: What does it mean?

### 2. Introduction
- **Background**: Contextualize the study.
- **Problem Statement**: What is the gap?
- **Hypothesis/Objective**: What are you testing?
- **Significance**: Why is this important?

### 3. Methods (Reproducibility)
- **Study Design**: Experimental setup, observational, etc.
- **Participants/Materials**: Who or what was studied?
- **Procedures**: Step-by-step execution.
- **Data Analysis**: Statistical tests and software used.

### 4. Results (Objectivity)
- **Key Findings**: Present data without interpretation.
- **Visuals**: Use tables and figures to summarize complex data.
- **No Speculation**: Save the "why" for the Discussion.

### 5. Discussion (Interpretation)
- **Summary**: Restate main findings.
- **Interpretation**: What do the results mean?
- **Context**: Compare with previous literature (support or contradict?).
- **Limitations**: Be honest about weaknesses.
- **Implications**: How does this advance the field?

### 6. Conclusion (The "So What?")
- **Restate Thesis**: Remind the reader of the main point.
- **Broader Impact**: Why does this matter to the world?
- **Future Directions**: What should be researched next?

---

## 💬 Inspiring Quotes for Researchers

> "Research is seeing what everybody else has seen and thinking what nobody else has thought."
> — *Albert Szent-Györgyi*

> "If we knew what we were doing, it wouldn't be called research, would it?"
> — *Albert Einstein*

> "Everything is theoretically impossible, until it is done."
> — *Robert A. Heinlein*

> "The most exciting phrase to hear in science, the one that heralds new discoveries, is not 'Eureka!' but 'That's funny...'"
> — *Isaac Asimov*

> "Nothing in life is to be feared, it is only to be understood. Now is the time to understand more, so that we may fear less."
> — *Marie Curie*

> "Without data, you're just another person with an opinion."
> — *W. Edwards Deming*

> "I am not a genius, I am just curious."
> — *Albert Einstein*


---


# Ruff Cheat Sheet (with uv)

Modern Python formatting and linting using `ruff` and `uv`.

## 1. Installation & Usage with uv

Since `uv` and `ruff` are both from Astral, they work together seamlessly.

### Add Ruff to your project
Add `ruff` as a development dependency:
```bash
uv add --dev ruff
```

### Basic Commands
Run ruff through `uv` to ensure you use the project's pinned version.

- **Lint (check errors):**
  ```bash
  uv run ruff check .
  ```
- **Lint (fix errors):**
  ```bash
  uv run ruff check --fix .
  ```
- **Format (check):**
  ```bash
  uv run ruff format --check .
  ```
- **Format (auto-format):**
  ```bash
  uv run ruff format .
  ```

### Using without installation (`uvx`)
You can run ruff instantly without installing it in the project:
```bash
uvx ruff check .
```

---

## 2. Configuration (`pyproject.toml`)

Centralize all configuration in `pyproject.toml`.

```toml
[project]
# ... standard project metadata ...

[tool.ruff]
# Target Python version (adjust to your project's needs)
target-version = "py310"
# Maximum line length (Ruff defaults to 88, similar to Black)
line-length = 88

[tool.ruff.lint]
# Enable rules:
# E: Pycodestyle errors
# F: Pyflakes
# I: isort (Import sorting)
# B: Flake8-bugbear (Potential bugs)
# UP: Pyupgrade (Upgrade syntax to newer Python versions)
# N: Pep8-naming
select = ["E", "F", "I", "B", "UP", "N"]

# Ignore specific rules if needed
ignore = []

# Allow unused variables when underscore-prefixed.
dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"

[tool.ruff.lint.isort]
# Organize imports settings
known-first-party = ["my_project"]

[tool.ruff.format]
# Like Black, use double quotes for strings.
quote-style = "double"
# Indent with spaces, rather than tabs.
indent-style = "space"
# Respect magic trailing commas.
skip-magic-trailing-comma = false
# Automatically detect the appropriate line ending.
line-ending = "auto"
```

---

## 3. VSCode Configuration

To use Ruff effectively in VSCode, install the **Ruff** extension (Extension ID: `astral-sh.ruff` or formerly `charliermarsh.ruff`).

Add the following to your `.vscode/settings.json` (workspace settings) or global `settings.json`:

```json
{
  "[python]": {
    "editor.defaultFormatter": "astral-sh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  // If you are using a custom venv or uv, Ruff usually finds it automatically.
  // But you can explicitly set the python path if needed (rarely required with uv python finding):
  // "ruff.interpreter": ["/path/to/.venv/bin/python"]
}
```
*Note: Ensure "Editor: Default Formatter" is set to Ruff for Python files.*

---

## 4. Advanced Formatting & Linting Usage

### Type Hints (mypy / flake8-type-checking equivalent)
Ruff doesn't do full type checking (use `mypy` or `pyright` for that), but it can enforce type hint strictness via `flake8-type-checking` rules (`TCH`).

**Configuration:**
```toml
[tool.ruff.lint]
# Add "TCH" to select
select = [..., "TCH"]
```

**What it does:**
It ensures imports used ONLY for type hinting are moved to a `TYPE_CHECKING` block to avoid circular imports and runtime costs.

**Example Code:**

*Before (Ruff flags this):*
```python
import pandas as pd  # Used only for type hint

def process_data(df: pd.DataFrame) -> None:
    pass
```

*After (`ruff check --fix`):*
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

def process_data(df: "pd.DataFrame") -> None:
    pass
```

### Docstrings (`pydocstyle` equivalent)
Enforce docstring standards (e.g., Google, NumPy, or PEP 257 style).

**Configuration:**
```toml
[tool.ruff.lint]
# Add "D" to select
select = [..., "D"]

[tool.ruff.lint.pydocstyle]
convention = "google"  # or "numpy", "pep257"
```

**Example Code:**

*Bad Docstring (Ruff flags missing args):*
```python
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b
```

*Good Docstring (Google Style):*
```python
def add(a: int, b: int) -> int:
    """Adds two numbers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b
```

### Import Sorting (isort equivalent)
Ruff has a built-in sort (Rule `I`). It sorts standard library, third-party, and local imports automatically.

**Configuration:**
Already included if you add `"I"` to `select`.

**Example:**
*Before:*
```python
import sys
from my_lib import foo
import os
import requests
```

*After:*
```python
import os
import sys

import requests

from my_lib import foo
```

### Ignoring Rules Inline
You can ignore specific rules for a line using `# noqa`.

```python
x = 1  # noqa: F841 (Variables assigned but never used)
```


---


# Scikit-learn Cheatsheet: Specialized Applications

This cheatsheet covers real-world application examples and specialized techniques demonstrated in the scikit-learn gallery.

## Domain-Specific Applications

### 1. Computer Vision & Face Recognition
- **Technique**: Eigenfaces (PCA) + Classification (SVM).
- **Goal**: Identify individuals from images.
- **Key Function**: `sklearn.decomposition.PCA`, `sklearn.svm.SVC`.

### 2. Finance & Ecology
- **Stock Market Embedding**: Discovering structure in the stock market using `GraphicalLassoCV` to learn the covariance matrix of price fluctuations.
- **Species Distribution**: Modeling the geographic distribution of species using `OneClassSVM` or `RandomForest`.

### 3. Text Mining & Topic Modeling
- **Topics Extraction**: Uncovering latent themes in document collections.
- **Key Function**: `sklearn.decomposition.NMF`, `sklearn.decomposition.LatentDirichletAllocation`.

## Advanced Engineering Techniques

### 4. Out-of-Core Classification
- **Scenario**: Dataset too large to fit in memory (Incremental Learning).
- **Key Function**: `partial_fit()` method available in models like `SGDClassifier`, `MultinomialNB`, and `HashingVectorizer`.

### 5. Time Series Forecasting
- **Technique**: Lagged features and cyclical engineering (sine/cosine transforms).
- **Goal**: Convert time-based data into features suitable for regression.

### 6. Outlier & Novelty Detection
- **Technique**: Identifying "weird" samples in a dataset.
- **Key Function**: `sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `sklearn.svm.OneClassSVM`.

## Computational Considerations
- **Prediction Latency**: Using `plot_prediction_latency` to analyze how different models scale with the number of features.
- **Model Complexity**: Analyzing the trade-off between model complexity (e.g., number of trees) and accuracy.

## Code Snippet: Out-of-Core Learning

```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

# HashingVectorizer is stateless and memory-efficient
vectorizer = HashingVectorizer(stop_words='english')
clf = SGDClassifier(loss='log_loss')

# Simulated stream of data
for X_batch_raw, y_batch in stream_generator():
    X_batch = vectorizer.transform(X_batch_raw)
    # partial_fit allows learning from data bit by bit
    clf.partial_fit(X_batch, y_batch, classes=all_classes)
```

## Code Snippet: Cyclical Feature Engineering (for Time Series)

```python
import numpy as np
import pandas as pd

# Convert hour of day (0-23) into cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Biclustering

Biclustering (also known as co-clustering or two-mode clustering) is a data mining technique which allows simultaneous clustering of the rows and columns of a matrix.

## What can be done?
- **Simultaneous Clustering**: Find blocks within a data matrix where rows and columns exhibit similar patterns.
- **Pattern Discovery**: Identify localized sub-structures that global clustering (like K-Means on just rows or just columns) might miss.
- **Dimensionality Reduction**: Focus on relevant sub-matrices in high-dimensional data.

## Algorithms in scikit-learn
1. **`SpectralCoclustering`**:
   - Finds biclusters with values higher than those in corresponding other rows and columns.
   - Typically used for document-word clustering (identifying topics and their associated documents).
2. **`SpectralBiclustering`**:
   - Assumes a checkerboard structure.
   - Normalizes data to make the checkerboard pattern apparent.

## Theoretical Background
Biclustering treats the data matrix as a bipartite graph. Applications of **Spectral Graph Theory** (specifically Singular Value Decomposition - SVD) are used to find optimal partitions.
- **Bipartite Graph**: One set of nodes for rows, another for columns. Edges represent matrix entries.
- **SVD**: Used to find the "spectrum" of the graph, helping to partition nodes (rows/columns) into clusters.

## Computational Complexity
- **Overall**: Generally $O(k \cdot min(m, n)^2)$ or higher depending on the implementation, where $k$ is number of clusters.
- **SVD**: The bottleneck is often the SVD step, which is $O(min(m^2n, mn^2))$ for a dense $m \times n$ matrix, though scikit-learn uses efficient iterative solvers (like Arnoldi or Randomized SVD).

## Application Examples
- **Bioinformatics**: Clustering genes and experimental conditions (finding genes that are co-expressed under specific conditions).
- **Text Mining**: Clustering documents and terms (finding specific vocabularies associated with document clusters).
- **E-commerce**: Clustering users and products (finding groups of users who like specific sets of items).

## Pros & Cons
### Pros
- **Local Patterns**: Can find clusters that only exist in a subset of features/samples.
- **Interpretability**: Provides direct links between row clusters and column clusters.
### Cons
- **Complexity**: More computationally expensive than simple clustering.
- **Heuristic**: Often requires specifying the number of clusters in advance.
- **Evaluation**: Hard to evaluate without ground truth (lack of standard internal metrics).

## Code Snippet

```python
import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_biclusters

# 1. Generate sample data
data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=5, shuffle=True, random_state=0
)

# 2. Fit the model
# n_clusters: The number of biclusters to find.
model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)

# 3. Access results
# model.row_labels_: Which row cluster each row belongs to
# model.column_labels_: Which column cluster each column belongs to
# model.biclusters_: Boolean mask (n_clusters, n_rows, n_cols)

# Rearranging data to visualize biclusters
fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Probability Calibration

Probability calibration is the process of adjusting the predicted probabilities of a classifier so they better reflect the actual likelihood of an event.

## What can be done?
- **Confidence Rating**: Transform raw model scores into reliable probability estimates (e.g., if a model says 80% probability, it should be correct 80% of the time).
- **Model Comparison**: Use Calibration Curves (Reliability Diagrams) to see which models are overconfident or underconfident.
- **Improved Decision Making**: Essential for risk-sensitive applications like medical diagnosis or financial forecasting.

## Calibration Methods
1. **Platt Scaling (`method='sigmoid'`)**:
   - Fits a logistic regression model to the classifier's outputs.
   - Best for Support Vector Machines (SVM) and when training data is small.
2. **Isotonic Regression (`method='isotonic'`)**:
   - Fits a non-parametric, non-decreasing function.
   - More powerful than Platt scaling but requires more data (~1000+ samples) and is prone to overfitting.

## Theoretical Background
- **Perfect Calibration**: A model's predicted probability equals the observed frequency of the positive class.
- **Brier Score**: A common metric to evaluate calibration. It measures the mean squared difference between predicted probability and actual outcome.
- **Cross-Validation**: Calibration should be performed on data not used for training the base estimator to avoid optimistic biases. `CalibratedClassifierCV` handles this automatically.

## Computational Complexity
- **Training**: Adds the cost of fitting a 1D regressor (Sigmoid or Isotonic).
- **Memory**: Minimal, as it only stores parameters for the mapping function.
- **Latency**: Negligible impact on prediction speed.

## Pros & Cons
### Pros
- **Interpretability**: Makes class probabilities meaningful.
- **Trust**: Allows users to know when the model is "unsure".
### Cons
- **Doesn't improve Accuracy**: Calibration adjusts probabilities but generally doesn't change the classification boundary or ROC-AUC.
- **Data Hungry**: Isotonic regression needs a decent amount of validation data.
- **Complexity**: Adds another step to the training pipeline.

## Code Snippet

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 1. Base Estimator (e.g., SVM which is notoriously uncalibrated)
base_clf = SVC(kernel='linear', C=1.0)

# 2. Wrap with Calibration
# method='sigmoid' (Platt) or 'isotonic'
calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

# 3. Predict Probabilities
probs = calibrated_clf.predict_proba(X_test)[:, 1]

# 4. Evaluation: Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Classification

Classification is a supervised learning task where the goal is to predict the categorical class of new observations.

## What can be done?
- **Binary & Multiclass Classification**: Predict between two or more discrete categories.
- **Decision Boundary Analysis**: Understand how models partition the feature space.
- **Dimensionality Reduction for Class Separation**: Use techniques like LDA to maximize class separability.

## Key Algorithms
1. **`LinearDiscriminantAnalysis` (LDA)**:
   - Finds a linear combination of features that separates classes.
   - Assumes classes follow a Gaussian distribution with shared covariance.
2. **`QuadraticDiscriminantAnalysis` (QDA)**:
   - Like LDA but allows each class to have its own covariance matrix.
   - Results in quadratic decision boundaries.
3. **Common Others** (Detailed in their own sections):
   - `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GaussianNB`.

## Theoretical Background
- **Bayes' Rule**: Many classifiers (LDA, QDA, Naive Bayes) are based on modeling the class conditional density $P(X|y)$.
- **Discriminative vs Generative**: 
  - Generative (LDA, QDA, NB): Models $P(X|y)$ and $P(y)$.
  - Discriminative (Logistic Regression, SVM): Models $P(y|X)$ directly.

## Computational Complexity
- **LDA**: $O(n \cdot p^2 + p^3)$ where $n=$ samples, $p=$ features.
- **Prediction**: Usually very fast ($O(p)$ for linear models).

## Evaluation Metrics
- **Accuracy**: $(TP + TN) / Total$.
- **Precision/Recall/F1**: For imbalanced classes.
- **ROC-AUC**: Ability to distinguish between classes across thresholds.
- **Confusion Matrix**: Detailed look at where types of errors occur.

## Code Snippet: Classifier Comparison

```python
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report

X, y = load_your_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

print("LDA Report:\n", classification_report(y_test, y_pred_lda))
print("QDA Report:\n", classification_report(y_test, y_pred_qda))
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Clustering

Clustering is an unsupervised learning task that groups a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups.

## What can be done?
- **Data Partitioning**: Group data into $K$ clusters (e.g., K-Means).
- **Density Discovery**: Find arbitrarily shaped clusters based on density (e.g., DBSCAN).
- **Hierarchy Building**: Create a tree of clusters (Agglomerative).
- **Anomalies Identification**: Points that don't fit into any cluster (noise in DBSCAN/OPTICS).

## Key Algorithms
1. **`KMeans`**:
   - Simplest and most common. Minimizes within-cluster sum-of-squares.
   - **MiniBatchKMeans**: Faster version for large datasets.
2. **`DBSCAN` / `HDBSCAN`**:
   - Density-based. Can find non-spherical clusters and marked outliers.
3. **`AgglomerativeClustering`**:
   - Hierarchical. Can incorporate "connectivity constraints" to group only adjacent points.
4. **`MeanShift`**:
   - Centroid-based. Finds peaks in a distribution; chooses number of clusters automatically.
5. **`AffinityPropagation`**:
   - Based on message passing between data points.

## Evaluation Metrics (Internal)
- **Silhouette Coefficient**: Measures how similar a point is to its own cluster compared to others. Higher (near 1) is better.
- **Calinski-Harabasz Index**: Ratio of between-cluster variance to within-cluster variance.
- **Davies-Bouldin Index**: Average "similarity" between clusters. Lower is better.

## Computational Complexity
- **K-Means**: $O(T \cdot K \cdot n \cdot p)$ (T: iterations, K: clusters, n: samples, p: features).
- **DBSCAN**: $O(n \cdot \log n)$ with spatial index or $O(n^2)$ without.
- **Agglomerative**: $O(n^2 \cdot \log n)$ to $O(n^3)$.

## Code Snippet: Clustering & Evaluation

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Scaling is CRITICAL for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-Means
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
print("K-Means Silhouette:", silhouette_score(X_scaled, labels_kmeans))

# 2. DBSCAN
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Compose (Pipelines & Meta-Estimators)

The `compose` module provides tools to combine multiple estimators into a single one, facilitating complex workflows and preventing data leakage.

## What can be done?
- **Chaining Steps**: Sequentially apply transformers and a final estimator.
- **Heterogeneous Data**: Apply different transformations to different columns (e.g., one-hot encoding for categorical, scaling for numerical).
- **Parallel Processing**: Combine multiple feature extraction methods.
- **Target Transformation**: Transform the target variable (e.g., log transform) automatically during fit/predict.

## Key Tools
1. **`Pipeline`**:
   - Chains multiple steps. Only the last step can be an estimator (model), others must be transformers.
   - Ensures that transformers are fitted on training data and applied to test data correctly.
2. **`ColumnTransformer`**:
   - Routes specific columns to specific transformers.
   - Essential for handling mixed-type data (numeric + categorical).
3. **`FeatureUnion`**:
   - Concatenates the results of multiple transformer objects.
4. **`TransformedTargetRegressor`**:
   - Wraps a regressor to apply a transformation to the target $y$ before fitting and an inverse transformation after predicting.

## Best Practices
- **Prevent Data Leakage**: Always use Pipelines when performing cross-validation to ensure preprocessing parameters (like mean/std) are calculated only on the training folds.
- **Hyperparameter Tuning**: You can tune hyperparameters of any step in a pipeline using `stepname__parametername` syntax in `GridSearchCV`.

## Code Snippet: Pipeline & ColumnTransformer

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 1. Define transformers for different column types
numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_features = ["embarked", "sex"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# 2. Bundle them in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 3. Create the final Pipeline
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

# 4. Use it as a single estimator
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Covariance Estimation

Covariance estimation is used to understand the relationship between variables and is a core component of many algorithms like LDA and Mahalanobis distance.

## What can be done?
- **Relationship Discovery**: Calculate how variables change together.
- **Outlier Detection**: Use Mahalanobis distance to find samples that deviate from the distribution.
- **Structure Learning**: Find sparse relationships where most variables are conditionally independent (graphical models).

## Key Algorithms
1. **`EmpiricalCovariance`**:
   - Standard Maximum Likelihood Estimator.
   - Unbiased but can be inaccurate (high variance) when the number of features is large relative to samples.
2. **Shrinkage Methods (`LedoitWolf`, `OAS`)**:
   - Mix empirical covariance with a simple target (like identity matrix).
   - `LedoitWolf`: Optimally calculates the shrinkage coefficient.
   - `OAS`: Similar to Ledoit-Wolf but designed for small sample sizes.
3. **`GraphicalLasso`**:
   - Learns a sparse inverse covariance matrix (precision matrix) using L1 penalty.
   - Useful for discovering network structures (which variables depend on which).
4. **`MinCovDet` (Minimum Covariance Determinant)**:
   - Robust estimator that ignores outliers.
   - Highly recommended for outlier detection data cleaning.

## Computational Complexity
- **Empirical**: $O(p^2 \cdot n)$ ( $n$: samples, $p$: features).
- **GraphicalLasso**: $O(p^3)$ or higher depending on convergence.

## Code Snippet: Robust vs Empirical Covariance

```python
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# Generate data with outliers
X = np.random.randn(100, 2)
X[0] = [10, 10] # Outlier

# 1. Standard Estimation (skewed by outlier)
emp_cov = EmpiricalCovariance().fit(X)
print("Empirical Location:", emp_cov.location_)

# 2. Robust Estimation (ignores outlier)
robust_cov = MinCovDet().fit(X)
print("Robust Location:", robust_cov.location_)

# 3. Get Mahalanobis Distances to detect outliers
distances = robust_cov.mahalanobis(X)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Cross Decomposition

Cross decomposition algorithms find fundamental relations between two matrices (X and Y). They are "latent variable" methods.

## What can be done?
- **Multi-output Regression**: Predict multiple dependent variables simultaneously.
- **Dimensionality Reduction**: Project X and Y into a lower-dimensional subspace where they are highly correlated.
- **PCR vs PLS**: PLS is often superior to Principal Component Regression (PCR) because it considers the relationship between X and Y when finding the directions of maximum variance.

## Key Algorithms
1. **`PLSRegression` (Partial Least Squares)**:
   - The most popular method. Finds directions in X that explain variance in X and are correlated with Y.
   - Great for datasets where the number of predictors exceeds the number of observations (high $p$, low $n$).
2. **`CCA` (Canonical Correlation Analysis)**:
   - Finds linear combinations of X and Y that have maximum correlation with each other.
   - Focuses purely on correlation, not variance.
3. **`PLSCanonical`**:
   - A variant of PLS that is more similar to CCA but uses a different objective function.

## Theoretical Background
- **Latent Variables**: Both X and Y are assumed to be generated by underlying, unobserved factors.
- **Iterative Deflation**: Algorithms often work by finding the first pair of latent vectors, then subtracting their contribution ("deflating") and finding the next.

## Computational Complexity
- Generally efficient, dominated by Singular Value Decomposition (SVD) or Eigendecomposition on small covariance matrices. Usually $O(n \cdot p^2)$.

## Code Snippet: PLS Regression

```python
from sklearn.cross_decomposition import PLSRegression
import numpy as np

# Multi-target data (X: 100x10, Y: 100x2)
X = np.random.randn(100, 10)
Y = np.random.randn(100, 2)

# Fit PLS
# n_components: Number of latent variables to keep
pls = PLSRegression(n_components=2)
pls.fit(X, Y)

# Transform X into latent space
X_scores, Y_scores = pls.transform(X, Y)

# Predict new values
Y_pred = pls.predict(X_new)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Datasets

The `datasets` module provides utilities to load standard datasets, fetch data from external repositories, and generate synthetic data for benchmarking and testing.

## What can be done?
- **Quick Prototyping**: Use "toy" datasets like Iris or Digits.
- **Benchmarking**: Fetch large real-world datasets from OpenML.
- **Algorithm Testing**: Generate synthetic datasets with controlled noise, clusters, or informative features.

## Categories of Datasets
1. **Toy Datasets**: Loaded immediately with the library.
   - `load_iris()`, `load_digits()`, `load_wine()`, `load_breast_cancer()`.
2. **Real-world Datasets**: Downloaded on demand.
   - `fetch_20newsgroups()`, `fetch_lfw_people()` (faces), `fetch_california_housing()`.
3. **OpenML**: Access thousands of datasets from openml.org.
   - `fetch_openml(name='mnist_784')`.
4. **Synthetic Generators**:
   - `make_classification()`: For binary/multiclass problems.
   - `make_regression()`: For regression problems.
   - `make_blobs()`: For clustering.
   - `make_moons()`, `make_circles()`: For non-linear separation tests.

## Tips
- **`as_frame=True`**: Many loaders have this argument to return data as a `pandas.DataFrame` instead of a NumPy array.
- **`return_X_y=True`**: Returns `(X, y)` directly instead of a `Bunch` object.

## Code Snippet: Generating and Loading Data

```python
from sklearn.datasets import make_classification, load_iris, fetch_openml

# 1. Load Toy Dataset as DataFrame
iris = load_iris(as_frame=True)
df = iris.frame

# 2. Generate Synthetic Classification Data
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    random_state=42
)

# 3. Fetch from OpenML
# version='active' or specific number
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Decomposition

Matrix decomposition techniques are used for dimensionality reduction, feature extraction, and signal separation.

## What can be done?
- **Dimensionality Reduction**: Reduce feature space while preserving variance (PCA).
- **Feature Extraction**: Identify latent patterns (NMF, Factor Analysis).
- **Blind Source Separation**: Separate mixed signals (ICA).
- **Denoising**: Reconstruct data by projecting onto principal components.

## Key Algorithms
1. **`PCA` (Principal Component Analysis)**:
   - Linearly transforms data to a new coordinate system of orthogonal axes (Principal Components).
   - **IncrementalPCA**: Processes data in batches (out-of-core).
   - **KernelPCA**: Uses kernels to find non-linear principal components.
2. **`NMF` (Non-negative Matrix Factorization)**:
   - All values in the decomposition are non-negative.
   - Ideal for part-based representations (e.g., topics in text, parts of faces).
3. **`FastICA` (Independent Component Analysis)**:
   - Finds components that are maximally independent (not just uncorrelated).
   - Famous for the "Cocktail Party Problem" (separating voices).
4. **`DictionaryLearning`**:
   - Learns a "dictionary" (basis vectors) such that data can be represented as sparse combinations of these vectors.

## Theoretical Background
- **SVD**: Singular Value Decomposition is the underlying engine for most PCA variants.
- **Variance Explained**: A key metric in PCA to decide how many components to keep.

## Computational Complexity
- **PCA**: $O(min(n^2 p, n p^2))$ for exact solver. Randomized PCA is much faster for high-dimensional data.

## Code Snippet: PCA & NMF

```python
from sklearn.decomposition import PCA, NMF
from sklearn.datasets import load_digits

X, _ = load_digits(return_X_y=True)

# 1. Standard PCA
# n_components can be an int or a float (ratio of variance to keep)
pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X)
print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} features")

# 2. NMF (only works with non-negative data)
nmf = NMF(n_components=10, init='random', random_state=0)
X_nmf = nmf.fit_transform(X)

# 3. ICA for signal separation
from sklearn.decomposition import FastICA
ica = FastICA(n_components=2)
S_source = ica.fit_transform(X_signals) # X_signals is mixed
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Developing Estimators

Scikit-learn allows you to create custom estimators that work seamlessly with `Pipeline`, `GridSearchCV`, and other utilities.

## What can be done?
- **Custom Transformers**: Implement specialized feature engineering.
- **Custom Models**: Wrap external libraries or unique algorithms into scikit-learn interface.
- **Enforcement**: Ensure your code follows the "Scikit-Learn API contract".

## Key Components
1. **Base Classes**:
   - `BaseEstimator`: Provides `get_params` and `set_params`.
   - `TransformerMixin`: Provides `fit_transform`.
   - `ClassifierMixin`: Provides `score` (accuracy) and sets `_estimator_type`.
   - `RegressorMixin`: Provides `score` (R2).
2. **Validation Utilities**:
   - `check_X_y`: Ensures data format and target are consistent.
   - `check_array`: Standardizes input array (handling NaN, types, etc.).
   - `check_is_fitted`: Raises an error if the model hasn't been fit yet.

## The Contract
- **`__init__`**: Must NOT have side effects. Should only assign arguments to attributes. No logic!
- **`fit(X, y)`**: Must return `self`.
- **`transform(X)`** or **`predict(X)`**: Perform the actual work after fitting.

## Code Snippet: Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class MyLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_constant=1.0):
        # Store parameters (no logic here!)
        self.add_constant = add_constant

    def fit(self, X, y=None):
        # Validate data
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        # Return self
        return self

    def transform(self, X):
        # Ensure it was fit
        check_is_fitted(self)
        X = check_array(X)
        # Apply transformation
        return np.log(X + self.add_constant)

# Validation tests
from sklearn.utils.estimator_checks import check_estimator
# check_estimator(MyLogTransformer()) # Runs many automated tests
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Ensemble Methods

Ensemble methods combine the predictions of several base estimators to improve generalizability and robustness over a single estimator.

## What can be done?
- **Reduce Overfitting**: Averaging techniques (Bagging).
- **Increase Accuracy**: Iterative techniques (Boosting).
- **Combine Heterogeneous Models**: Voting and Stacking.
- **Quantile Regression**: Use Gradient Boosting to predict intervals.

## Key Algorithms
1. **Bagging (Averaging)**:
   - `RandomForestClassifier/Regressor`: Builds many deep trees on subsets of data/features and averages results.
2. **Boosting (Sequential)**:
   - `GradientBoostingClassifier/Regressor`: Fits new models to the residuals of previous models.
   - `HistGradientBoosting`: Modern, extremely fast version similar to LightGBM.
   - `AdaBoost`: Focuses more on samples that previous models misclassified.
3. **Voting**:
   - `VotingClassifier`: Combines different models via majority vote (hard) or average probability (soft).
4. **Stacking**:
   - `StackingClassifier`: Trains a "final estimator" (meta-learner) to combine predictions of base learners.

## Theoretical Background
- **Bias-Variance Trade-off**: Bagging reduces variance (good for complex models like deep trees). Boosting reduces bias (good for weak models like shallow trees).
- **Out-of-Bag (OOB) Score**: Validation method for Random Forest using samples not seen by specific trees.

## Computational Complexity
- **Random Forest**: Parallelizable. $O(M \cdot n \cdot \log n)$ where $M$ is number of trees.
- **Gradient Boosting**: Sequential (harder to parallelize, except Hist).

## Code Snippet: Random Forest & HistGradientBoosting

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# 1. Random Forest (Great baseline)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
# n_jobs=-1 uses all CPU cores

# 2. HistGradientBoosting (Fast for large datasets)
# Categorical support is built-in!
hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)

# 3. Voting Ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC(probability=True)

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], 
    voting='soft'
)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Feature Selection

Feature selection reduces the number of input variables to reduce overfitting, improve accuracy, and decrease computational cost.

## What can be done?
- **Filter Methods**: Use statistical scores to pick features.
- **Wrapper Methods**: Search for the best subset of features using a model.
- **Embedded Methods**: Let models (like Lasso or Random Forest) select features during training.

## Key Algorithms
1. **`SelectKBest` (Filter)**:
   - Selects high-scoring features based on `f_classif` (ANOVA), `chi2`, or `mutual_info_classif`.
2. **`RFE` (Recursive Feature Elimination - Wrapper)**:
   - Iteratively fits a model and removes the least important features.
   - **`RFECV`**: Automatically finds the optimal number of features using cross-validation.
3. **`SelectFromModel` (Embedded)**:
   - Uses `feature_importances_` or `coef_` attributes of a fitted estimator.
   - Works well with `Lasso` (L1 penalty) or `RandomForest`.
4. **`SequentialFeatureSelector`**:
   - Greedy forward or backward selection. More robust but slower than RFE.

## Theoretical Background
- **Mutual Information**: Captures any kind of statistical dependency (nonlinear), whereas F-test/Chi2 only capture linear relationships.
- **L1 Regularization (Lasso)**: Forces coefficients to be exactly zero, performing automatic selection.

## Computational Complexity
- **Filter**: Very fast.
- **RFE**: Slow, as it fits the model multiple times ($O(p)$ times).

## Code Snippet: Feature Selection Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# 1. Filter: Select top 10 features via ANOVA
selector = SelectKBest(score_func=f_classif, k=10)

# 2. Wrapper: Recursive Feature Elimination
# Requires an estimator that provides feature importance (e.g. RF, Linear)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)

# 3. Embedded: Select from Model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
sfm = SelectFromModel(Lasso(alpha=0.1))

# Integrating into Pipeline
pipe = Pipeline([
    ('feature_selection', selector),
    ('classification', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Frozen Estimators

A `FrozenEstimator` is a wrapper that prevents a fitted estimator from being updated during a subsequent `fit` call.

## What can be done?
- **Transfer Learning**: Use a pre-trained model as a fixed feature extractor in a larger pipeline.
- **Multi-stage Training**: Fit one part of a model, freeze it, and then fit a second part that depends on the first.
- **Resource Optimization**: Avoid re-fitting expensive models when only a small part of the pipeline changes.

## Key Concept
- The `FrozenEstimator` delegates `predict`, `transform`, and other methods to the underlying estimator but its `fit` method does nothing (effectively "no-op").

## Code Snippet: Freezing an Estimator

```python
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Fit an estimator normally
clf = LogisticRegression().fit(X_train, y_train)

# 2. Freeze it
frozen_clf = FrozenEstimator(clf)

# 3. Use in a new pipeline
# When pipe.fit() is called, frozen_clf.fit() does nothing.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', frozen_clf)
])
pipe.fit(X_new, y_new) 
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Gaussian Processes

Gaussian Processes (GP) are a generic supervised learning method designed to solve regression and probabilistic classification problems.

## What can be done?
- **Regression with Uncertainty**: Get not just a prediction, but a confidence interval ($ \mu \pm \sigma $).
- **Interpolation**: GPs can perfectly interpolate training points (if noise is set to zero).
- **Probabilistic Classification**: Get class probabilities based on the Gaussian process.

## Key Components
1. **`GaussianProcessRegressor`**:
   - Standard GP for regression.
2. **`GaussianProcessClassifier`**:
   - Uses a Laplace approximation for the non-Gaussian posterior.
3. **Kernels**:
   - **`RBF`**: Squared exponential kernel (smoothness).
   - **`Matern`**: Generalization of RBF (allows control over smoothness).
   - **`WhiteKernel`**: Models noise.
   - **`ExpSineSquared`**: Models periodic functions.
   - **`RationalQuadratic`**: Models mixtures of RBF kernels with different length scales.

## Theoretical Background
- **Bayesian Inference**: GPs define a prior over functions and update it with data to get a posterior.
- **Kernel Trick**: The kernel function defines the covariance between any two points in the feature space.

## Computational Complexity
- **Training**: $O(n^3)$ due to matrix inversion ( $n$: number of samples).
- **Inference**: $O(n^2)$ for variance, $O(n)$ for mean.
- **Limitation**: Not suitable for datasets with more than a few thousand samples without approximations.

## Code Snippet: GP Regression with Kernels

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# 1. Define Kernel
# C(1.0) * RBF(1.0) + WhiteKernel(noise_level=1)
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(1e-1)

# 2. Fit model
# n_restarts_optimizer: Run optimization multiple times to avoid local minima
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)

# 3. Predict with uncertainty
y_pred, sigma = gp.predict(X_new, return_std=True)
# y_pred: Mean of posterior, sigma: Standard deviation
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Impute (Handling Missing Data)

The `impute` module provides strategies to handle missing values (NaN) in datasets.

## What can be done?
- **Univariate Imputation**: Fill missing values using a single feature's statistics.
- **Multivariate Imputation**: Predict missing values using all other available features (more accurate).
- **Nearest Neighbor Imputation**: Fill based on similar samples.
- **Marking Missingness**: Add a binary indicator feature for where data was missing.

## Key Algorithms
1. **`SimpleImputer`**:
   - `strategy='mean'`, `'median'`, `'most_frequent'`, or `'constant'`.
   - Fast and simple baseline.
2. **`IterativeImputer`**:
   - Models each feature with missing values as a function of others in a round-robin fashion.
   - Inspired by R's MICE (Multivariate Imputation by Chained Equations).
3. **`KNNImputer`**:
   - Finds $K$ nearest neighbors for each sample with a missing value and averages their values for that feature.
4. **`MissingIndicator`**:
   - Useful when the *fact* that a value is missing is informative.

## Tips
- Always `fit` on training data and `transform` on test data.
- If using `IterativeImputer`, you must enable it first as it is experimental: `from sklearn.experimental import enable_iterative_imputer`.

## Code Snippet: Advanced Imputation Pipeline

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

X = [[1, 2], [np.nan, 3], [7, 6], [4, np.nan]]

# 1. Simple Mean Imputation
imp_mean = SimpleImputer(strategy='mean')
X_simple = imp_mean.fit_transform(X)

# 2. KNN Imputation (Weights by distance)
imp_knn = KNNImputer(n_neighbors=2, weights="distance")
X_knn = imp_knn.fit_transform(X)

# 3. Iterative Imputation
imp_iter = IterativeImputer(max_iter=10, random_state=0)
X_iter = imp_iter.fit_transform(X)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Inspection (Interpretability)

The `inspection` module provides tools to understand how a model makes decisions and which features are most important.

## What can be done?
- **Feature Importance**: Quantify how much each feature contributes to the prediction.
- **Partial Dependence**: Visualize the relationship between a target and a feature, averaging out others.
- **Decision Boundaries**: Plot how a classifier separates classes in a 2D plane.

## Key Tools
1. **`permutation_importance`**:
   - Measures importance by randomly shuffling a single feature and seeing how much the score drops.
   - **Advantage**: Model-agnostic and doesn't overemphasize high-cardinality features (unlike Random Forest's impurity-based importance).
2. **`PartialDependenceDisplay`**:
   - Shows the marginal effect one or two features have on the predicted outcome.
   - Helps identify if the relationship is linear, monotonic, or complex.
3. **`DecisionBoundaryDisplay`**:
   - Convenient tool to plot the class decision boundaries in a 2D feature space.

## Comparison: Global vs Local Importance
- **Impurity-based (RF)**: Fast, but biased toward features with many unique values.
- **Permutation-based**: Slower, but more reliable and works on any model.

## Code Snippet: Permutation Importance & PDP

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier().fit(X_train, y_train)

# 1. Permutation Importance
# n_repeats: number of times to shuffle each feature
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
# result.importances_mean contains the importance scores

# 2. Partial Dependence Plot (PDP)
# features: index or name of features to plot
display = PartialDependenceDisplay.from_estimator(clf, X, features=[0, 1, (0, 1)])
# (0, 1) plots a 2D interaction contour
```

## Code Snippet: Decision Boundary Display

```python
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

# Only works with 2 features (X has shape [n, 2])
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict", cmap=plt.cm.RdYlBu, alpha=0.8
)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Kernel Approximation

Kernel approximation allows you to use kernel-based methods (like SVM with RBF kernel) on large datasets by projecting features into a finite-dimensional space where linear models can be used.

## What can be done?
- **Scaling Kernels**: Use RBF, Polynomial, or Laplacian kernels on datasets too large for standard `SVC`.
- **Faster Training**: Transform features once and then use fast linear solvers like `SGDClassifier` or `Ridge`.
- **Precomputed Gram Matrices**: Approximate the kernel matrix when it cannot be stored in memory.

## Key Algorithms
1. **`RBFSampler`**:
   - Approximates the RBF kernel using Random Fourier Features.
   - Map features to a space where the dot product approximates the kernel $K(x, y) = \exp(-\gamma ||x-y||^2)$.
2. **`Nystroem`**:
   - Approximates a general kernel by using a subset of the training data (landmarks).
   - More accurate than `RBFSampler` for many datasets but requires keeping the training samples in memory for the transformation.
3. **`PolynomialCountSketch`**:
   - Approximates the polynomial kernel.

## Theoretical Background
- **Bochner's Theorem**: Any shift-invariant kernel (like RBF) is the Fourier transform of a positive measure. This is the basis for `RBFSampler`.
- **Low-rank Approximation**: `Nystroem` finds a low-rank approximation of the (potentially infinite) kernel matrix.

## Computational Complexity
- **Standard SVC**: $O(n^2)$ to $O(n^3)$.
- **Kernel Approx + Linear Model**: $O(n \cdot d^2)$ where $d$ is the number of components in the approximation.

## Code Snippet: Scaling RBF Kernel

```python
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# 1. Random Fourier Features
rbf_feature = RBFSampler(gamma=1, n_components=100, random_state=1)
X_features = rbf_feature.fit_transform(X)

# 2. Pipeline for Large Scale Learning
# This allows using "RBF-like" SVC on millions of rows
pipe = Pipeline([
    ('kernel_approx', Nystroem(kernel='rbf', n_components=300)),
    ('linear_model', SGDClassifier(loss='hinge')) # hing loss + kernel = SVM
])

pipe.fit(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Linear Models

Linear models assume that the target value is expected to be a linear combination of the input variables.

## What can be done?
- **Regression**: Predict continuous values (Price, Temperature).
- **Classification**: Predict categories (Spam/Not Spam).
- **Regularization**: Avoid overfitting by penalizing large coefficients.
- **Robust Regression**: Handle outliers in the data.

## Key Algorithms
1. **Regression**:
   - `LinearRegression`: Ordinary Least Squares (no penalty).
   - `Ridge`: L2 penalty (prevents large coefficients, handles multicollinearity).
   - `Lasso`: L1 penalty (forces coefficients to zero, performs feature selection).
   - `ElasticNet`: Combination of L1 and L2.
2. **Classification**:
   - `LogisticRegression`: Despite name, it's for classification.
   - `SGDClassifier`: Linear model optimized via Stochastic Gradient Descent (excellent for large data).
3. **Robust Regression**:
   - `RANSACRegressor`: Fits model by ignoring outliers.
   - `HuberRegressor`: Less sensitive to outliers than OLS.
4. **Bayesian**:
   - `BayesianRidge`: Ridge with automated hyperparameter estimation.

## Theoretical Background
- **Loss Functions**: OLS uses Mean Squared Error. Logistic Regression uses Log-Loss.
- **Regularization**: 
  - $L_1$: $\alpha \cdot ||w||_1$ (Sparsity)
  - $L_2$: $0.5 \cdot \alpha \cdot ||w||_2^2$ (Small weights)

## Computational Complexity
- **OLS**: $O(p^2 n + p^3)$ for $n$ samples, $p$ features.
- **SGD**: $O(k \cdot n \cdot p)$ ( $k$: iterations). Scale linearly with samples.

## Code Snippet: Lasso and Ridge

```python
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Regression with Regularization
# alpha: Strength of penalty (higher = more regularization)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# 2. Classification with L1 (Sparse features)
# solver='liblinear' or 'saga' required for L1
log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)

# 3. Large Scale learning
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Manifold Learning

Manifold learning is an approach to non-linear dimensionality reduction. It assumes that data lies along a low-dimensional "manifold" embedded in high-dimensional space.

## What can be done?
- **Visualization**: Project high-dimensional data (e.g., 64D images or 1000D embeddings) into 2D or 3D.
- **Non-linear Structure Recovery**: Find structures that PCA (linear) cannot see (e.g., Swiss Roll, S-curve).

## Key Algorithms
1. **`TSNE` (t-distributed Stochastic Neighbor Embedding)**:
   - Most popular for visualization. Keeps similar points together and dissimilar points apart.
   - **Note**: Not for feature engineering (output doesn't preserve global distances or scale).
2. **`Isomap`**:
   - Seeks a low-dimensional embedding that maintains "geodesic distances" between all points.
3. **`LLE` (Locally Linear Embedding)**:
   - Recovers global structure from locally linear fits.
4. **`MDS` (Multidimensional Scaling)**:
   - Aims to preserve the distances between points as much as possible.
5. **`SpectralEmbedding`**:
   - Uses Eigendecomposition of the graph Laplacian.

## Best Practices
- **Scale First**: Always use `StandardScaler` before manifold learning.
- **PCA Preprocessing**: For high-dimensional data, run PCA first (e.g., to 50D) before T-SNE to reduce noise and speed up computation.
- **Perplexity (T-SNE)**: Tune this hyperparameter (usually 30-50). It balances local vs global attention.

## Computational Complexity
- **T-SNE**: $O(n \log n)$ with Barnes-Hut, but can be slow for $n > 50,000$.
- **Isomap/MDS**: Generally $O(n^2)$ or $O(n^3)$.

## Code Snippet: T-SNE for Visualization

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# 1. Apply T-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# 2. Plotting (Visualization)
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10')
plt.colorbar()
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Miscellaneous Tools

This category contains various utility models and comparisons that don't fit into the main categories.

## Key Techniques

### 1. Isotonic Regression (`IsotonicRegression`)
- Fits a non-decreasing function to data.
- Non-parametric: No assumption about the functional form (e.g., linear vs quadratic).
- Useful for probability calibration and modeling monotonic relationships.

### 2. Anomaly & Outlier Detection Comparison
- Scikit-learn provides several algorithms, each with different strengths:
  - **`IsolationForest`**: Fast, based on tree ensembles. Best for high-dimensional data.
  - **`LocalOutlierFactor` (LOF)**: Based on local density. Good for finding outliers relative to their clusters.
  - **`OneClassSVM`**: Best for novelty detection when the training data contains only "normal" samples.

### 3. Dimensionality Reduction Bounds
- **`johnson_lindenstrauss_bound`**: A theoretical tool to estimate the number of dimensions required to preserve distances between points when using random projections.

### 4. Kernel Ridge Regression (`KernelRidge`)
- Combines Ridge regression (L2) with the kernel trick.
- Similar to SVR but uses squared error loss and has a closed-form solution.

## Code Snippet: Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 1. Isolation Forest (Global anomaly)
iso = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso.fit_predict(X) # -1 for anomaly, 1 for normal

# 2. Local Outlier Factor (Local anomaly)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)
```

## Code Snippet: Isotonic Regression

```python
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
y_fit = ir.fit_transform(x, y)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Mixture Models

Mixture models represent data as being generated from a mixture of several component distributions (usually Gaussian).

## What can be done?
- **Soft Clustering**: Get probabilities of a sample belonging to each cluster.
- **Density Estimation**: Model complex probability distributions as sums of Gaussians.
- **Outlier Detection**: Samples in very low-density regions of the mixture are outliers.

## Key Algorithms
1. **`GaussianMixture` (GMM)**:
   - Uses the Expectation-Maximization (EM) algorithm to fit a set of Gaussians.
   - You must specify the number of components $K$.
2. **`BayesianGaussianMixture`**:
   - A variant that integrates over parameters using Variational Inference.
   - Can automatically "zero out" unnecessary components, helping discover the true number of clusters.

## Covariance Types
GMM allows different constraints on the covariance matrices:
- `'full'`: Each component has its own general covariance (most flexible).
- `'tied'`: All components share the same general covariance.
- `'diag'`: Each component has its own diagonal covariance.
- `'spherical'`: Each component has its own single variance (simplest).

## Theoretical Background
- **EM Algorithm**: 
  - **E-step**: Estimate the probability (responsibility) of each sample for each Gaussian.
  - **M-step**: Update Gaussian parameters to maximize likelihood based on responsibilities.
- **BIC/AIC**: Criteria used to select the optimal number of components.

## Computational Complexity
- $O(k \cdot n \cdot p^2)$ where $n$: samples, $p$: features, $k$: components.

## Code Snippet: GMM Clustering

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# 1. Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 2. Predict labels (Hard clustering)
labels = gmm.predict(X)

# 3. Predict probabilities (Soft clustering)
probs = gmm.predict_proba(X)

# 4. Find optimal components via BIC
bic_scores = []
n_components_range = range(1, 10)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n).fit(X)
    bic_scores.append(gmm.bic(X))
# Best n is the one where BIC is lowest
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Model Selection & Evaluation

The `model_selection` module provides tools to split data, tune hyperparameters, and evaluate model performance.

## What can be done?
- **Robust Evaluation**: Use Cross-Validation to ensure the model generalizes.
- **Hyperparameter Tuning**: Find the best settings for your model (e.g., $C$ in SVM, $k$ in KNN).
- **Diagnostics**: Analyze underfitting vs. overfitting using learning curves.
- **Threshold Tuning**: Select the optimal decision threshold for classification.

## Key Tools
1. **Cross-Validation Splitters**:
   - `KFold`: Standard split.
   - `StratifiedKFold`: Preserves class proportions (essential for classification).
   - `TimeSeriesSplit`: Respects temporal order.
2. **Hyperparameter Search**:
   - `GridSearchCV`: Exhaustive search over specified parameter values.
   - `RandomizedSearchCV`: Samples from distributions (faster, often as good as grid search).
   - `HalvingGridSearch`: Efficient search using "successive halving" (early stopping for poor params).
3. **Visualization Displays**:
   - `LearningCurveDisplay`: Plots score vs. training set size.
   - `ValidationCurveDisplay`: Plots score vs. single hyperparameter.
   - `RocCurveDisplay`, `PrecisionRecallDisplay`.

## Best Practices
- **Nested Cross-Validation**: Use to get an unbiased estimate of the performance when performing hyperparameter tuning.
- **Scoring**: Specify `scoring='roc_auc'` or `scoring='f1'` to optimize for metrics other than accuracy.

## Code Snippet: Grid Search & Cross-Validation

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# 1. Setup Parameter Grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# 2. Setup Search
cv = StratifiedKFold(n_splits=5)
# refit=True: Fits the best model on the whole training set
grid = GridSearchCV(SVC(), param_grid, cv=cv, scoring='accuracy', refit=True)

# 3. Execution
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
best_model = grid.best_estimator_
```

## Code Snippet: Learning Curve

```python
from sklearn.model_selection import LearningCurveDisplay
import matplotlib.pyplot as plt

display = LearningCurveDisplay.from_estimator(
    estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5)
)
display.plot()
plt.show()
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Multiclass

Multiclass classification refers to those classification tasks that have more than two targets.

## What can be done?
- **Convert Binary to Multiclass**: Use meta-estimators to apply binary classifiers (like SVM) to multiclass problems.
- **Decision Strategy**: Choose between "One-vs-Rest" or "One-vs-One".

## Key Algorithms
1. **`OneVsRestClassifier` (OvR)**:
   - Fits one classifier per class.
   - Computational burden is $N$ classifiers.
   - Most common and scalable strategy.
2. **`OneVsOneClassifier` (OvO)**:
   - Fits one classifier for every pair of classes.
   - Computational burden is $N \cdot (N-1) / 2$ classifiers.
   - Often used for algorithms that don't scale well with the volume of data (like kernels).
3. **`OutputCodeClassifier`**:
   - Uses Error-Correcting Output Codes. Each class is represented by a binary code.

## Native Multiclass support
Some estimators handle multiclass natively (no wrapper needed):
- `LogisticRegression` (using `'multinomial'` loss).
- `RandomForestClassifier`.
- `GaussianNB`.
- `LinearDiscriminantAnalysis`.

## Code Snippet: Multiclass Strategies

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# 1. One-vs-Rest
ovr = OneVsRestClassifier(SVC(kernel='linear'))
ovr.fit(X, y)

# 2. One-vs-One
ovo = OneVsOneClassifier(SVC(kernel='linear'))
ovo.fit(X, y)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Multioutput

Multioutput (also known as multitarget or multiresponse) regression and classification involves predicting multiple target variables for each sample.

## What can be done?
- **Predict Multiple Values**: Simultaneously predict several continuous variables (regression) or categorical variables (classification).
- **Model Target Dependencies**: Use chains to let predictions for one target inform the prediction for the next.
- **Multilabel Classification**: Assign multiple labels to a single sample.

## Key Strategies
1. **`MultiOutputRegressor` / `MultiOutputClassifier`**:
   - Independently fits one regressor/classifier per target.
   - Simple baseline that assumes targets are independent.
2. **`RegressorChain` / `ClassifierChain`**:
   - Arranges estimators in a chain.
   - Target $i$ is predicted using the input features plus the predictions of targets $0, \dots, i-1$.
   - Captures correlations between targets.

## Data Format
- Target $y$ should be a 2D array of shape `(n_samples, n_targets)`.

## Code Snippet: Multioutput Regression & Chains

```python
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Multi-target data (X: 100x10, Y: 100x3)
X = np.random.randn(100, 10)
Y = np.random.randn(100, 3)

# 1. Independent Multioutput
mor = MultiOutputRegressor(RandomForestRegressor())
mor.fit(X, Y)

# 2. Regressor Chain (captures target dependencies)
chain = RegressorChain(RandomForestRegressor())
chain.fit(X, Y)
Y_pred = chain.predict(X_new)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Nearest Neighbors

Nearest Neighbors algorithms find a predefined number of training samples closest in distance to a new point.

## What can be done?
- **Supervised Learning**: Classification and Regression based on consensus of neighbors.
- **Unsupervised Learning**: Find neighbors for manifold learning or data analysis.
- **Density Estimation**: Estimate the probability distribution of data (KDE).
- **Anomaly Detection**: Find points that have low local density.

## Key Algorithms
1. **`KNeighborsClassifier/Regressor`**:
   - Standard KNN. Majority vote for classification, average for regression.
2. **`RadiusNeighborsClassifier`**:
   - Based on points within a fixed distance $r$. Good for data with varied density.
3. **`NearestNeighbors`**:
   - Unsupervised learner. Used for retrieving neighbors (`kneighbors()`) or graph building (`kneighbors_graph()`).
4. **`KernelDensity` (KDE)**:
   - Represents the data as a sum of kernels. Provides a smooth continuous density estimate.
5. **`LocalOutlierFactor` (LOF)**:
   - Compares local density of a point to its neighbors. High LOF score = Outlier.
6. **`NeighborhoodComponentsAnalysis` (NCA)**:
   - Learn a linear transformation to improve KNN accuracy.

## Computational Complexity
- **Brute Force**: $O[D \cdot N^2]$ (Too slow for large data).
- **KD-Tree / Ball-Tree**: $O[D \cdot N \log N]$ (Faster for low dimensions).
- **Inference**: Can be expensive for large $N$ since data must be kept.

## Code Snippet: KNN and KDE

```python
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
import numpy as np

# 1. KNN Classification
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
knn.fit(X_train, y_train)

# 2. Unsupervised Neighbor Search
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

# 3. Kernel Density Estimation
# bandwidth: The "width" of the kernels. Crucial parameter to tune!
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
log_density = kde.score_samples(X_plot)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Neural Networks

Scikit-learn provides basic neural network models (Multi-layer Perceptrons) and Restricted Boltzmann Machines.

## What can be done?
- **Supervised Learning**: Classification and Regression using deep architectures.
- **Feature Extraction**: Use RBMs to learn high-level features for a subsequent classifier.
- **Complex Non-linear Modeling**: MLPs can approximate any continuous function.

## Key Algorithms
1. **`MLPClassifier` / `MLPRegressor`**:
   - Multi-layer Perceptron (Vanilla Feedforward NN).
   - **Solvers**: 
     - `'adam'`: Default. Works well on large datasets.
     - `'lbfgs'`: Optimizer for small datasets (faster and more stable).
     - `'sgd'`: Standard stochastic gradient descent.
2. **`BernoulliRBM`**:
   - Unsupervised generative model with binary hidden and visible units.
   - Typically used in a pipeline before a linear classifier.

## Parameters to Tune
- `hidden_layer_sizes`: e.g., `(100, 50)` for two layers.
- `activation`: `'relu'` (default), `'logistic'` (sigmoid), `'tanh'`.
- `alpha`: L2 regularization parameter (to prevent overfitting).
- `early_stopping`: Set to `True` to stop training when validation score stalls.

## Computational Complexity
- $O(n \cdot m \cdot k \cdot o \cdot i)$ where $n$: samples, $m$: features, $k$: hidden units, $o$: output units, $i$: iterations.
- Much slower than linear models.

## Code Snippet: MLP & RBM

```python
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. Multi-layer Perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001,
    max_iter=500,
    random_state=1
)
mlp.fit(X_train, y_train)

# 2. RBM Feature Extraction + Logistic Regression
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=20)
logistic = LogisticRegression(C=100)

pipe = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
pipe.fit(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Preprocessing

Preprocessing is the transformation of raw data into a format that is more suitable for machine learning algorithms.

## What can be done?
- **Scaling**: Adjust the range or distribution of numeric features.
- **Encoding**: Convert categorical strings into numeric values.
- **Transformation**: Handle skewed data or non-linear relationships.
- **Discretization**: Bin continuous variables into intervals.

## Key Tools
1. **Scalers**:
   - `StandardScaler`: $z = (x - \mu) / \sigma$. Standardizes to mean 0, variance 1.
   - `MinMaxScaler`: Scales to range [0, 1]. Sensitive to outliers.
   - `RobustScaler`: Uses median and interquartile range (IQR). Best for outlier-heavy data.
2. **Encoders**:
   - `OneHotEncoder`: Binary column per category.
   - `OrdinalEncoder`: Maps categories to integers [0, 1, 2...].
   - `TargetEncoder`: Encode categories based on the target mean (useful for high-cardinality).
3. **Power Transforms**:
   - `PowerTransformer`: Box-Cox or Yeo-Johnson transforms to make data more Gaussian-like.
   - `QuantileTransformer`: Maps data to a uniform or normal distribution.
4. **Generating Features**:
   - `PolynomialFeatures`: Creates $x^2, xy, y^2$. Captures interactions between features.

## Tips
- **Fit on Train, Transform on Test**: Never fit a scaler on the whole dataset to avoid data leakage.
- **Standardization**: Required for distance-based models (KNN, SVM, K-Means) and Gradient Descent.

## Code Snippet: Common Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

# 1. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Interaction Terms
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# 3. Target Encoding (for categorical features)
from sklearn.preprocessing import TargetEncoder
encoder = TargetEncoder(smooth='auto')
X_encoded = encoder.fit_transform(X_cat, y)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Release Highlights (Modern Features)

Scikit-learn evolves rapidly. This cheatsheet highlights key features introduced in recent major versions (0.22 to 1.7+).

## Key Recent Milestones

### 1. Histogram-based GBDT (v0.21+)
- **`HistGradientBoostingClassifier/Regressor`**: Extremely fast gradient boosting, inspired by LightGBM. Supports native categorical features and missing values.

### 2. Interaction & Constraint Support
- **Monotonic Constraints**: Force a model (GBDT) to be always increasing or decreasing with respect to a feature.
- **Interaction Constraints**: Control which features are allowed to interact in tree splits.

### 3. Advanced Encoders & Transformers
- **`TargetEncoder` (v1.3+)**: Efficiently handles high-cardinality categorical features by using the target variable's mean. It includes smoothing to prevent overfitting.
- **`SplineTransformer` (v1.0+)**: For non-linear feature engineering, providing a better alternative to polynomial features in many cases.

### 4. Plotting & Displays API
- Unified API for plotting common charts: `RocCurveDisplay`, `ConfusionMatrixDisplay`, `PrecisionRecallDisplay`, `CalibrationDisplay`, `DecisionBoundaryDisplay`.

### 5. Native Categorical Support
- `HistGradientBoosting` can now handle categorical data directly without one-hot encoding if `categorical_features` parameter is set.

## Code Snippet: Modern Features

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import TargetEncoder

# 1. Native Categorical GBDT
hgb = HistGradientBoostingClassifier(categorical_features=[0, 3, 5])
hgb.fit(X, y)

# 2. Easy Confusion Matrix Plotting
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(hgb, X_test, y_test)

# 3. Target Encoding
te = TargetEncoder()
X_encoded = te.fit_transform(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Semi-Supervised Learning

Semi-supervised learning is used when you have a small amount of labeled data and a large amount of unlabeled data.

## What can be done?
- **Label Propagation**: Predict labels for the unlabeled portion of your dataset.
- **Self-Training**: Use a standard supervised model to iteratively label the unlabeled data.
- **Benefit**: Can significantly improve performance over purely supervised learning when labeling is expensive.

## Key Algorithms
1. **`SelfTrainingClassifier`**:
   - A wrapper that can use any classifier that provides `predict_proba`.
   - In each iteration, it adds the most confident predictions on unlabeled data to the training set.
2. **`LabelPropagation`**:
   - Uses a similarity graph between all data points to "spread" labels.
   - Based on the "Label Propagation" algorithm by Zhu and Ghahramani.
3. **`LabelSpreading`**:
   - Similar to LabelPropagation but uses a regularized graph Laplacian. 
   - Generally more robust to noise.

## Input Format
- Labels for unlabeled samples must be marked as `-1`.

## Theoretical Background
- **Manifold Assumption**: Points on the same low-dimensional manifold should have the same label.
- **Cluster Assumption**: Points in the same cluster are likely to have the same class.

## Code Snippet: Self-Training & Label Spreading

```python
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.svm import SVC

# 1. Mark unlabeled data as -1
y_train_mixed = np.copy(y_train)
random_unlabeled_indices = np.random.rand(len(y_train)) < 0.7
y_train_mixed[random_unlabeled_indices] = -1

# 2. Self-Training Wrapper
# threshold: minimum probability to include a sample in the next iteration
svc = SVC(probability=True, gamma="auto")
self_training_model = SelfTrainingClassifier(svc, threshold=0.75)
self_training_model.fit(X_train, y_train_mixed)

# 3. Label Spreading (Graph-based)
# kernel: 'knn' or 'rbf'
label_spread = LabelSpreading(kernel='knn', n_neighbors=7)
label_spread.fit(X_train, y_train_mixed)
predicted_labels = label_spread.transduction_
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Support Vector Machines (SVM)

Support Vector Machines are powerful supervised learning models used for classification, regression, and outlier detection.

## What can be done?
- **High-Dimensional Classification**: Extremely effective in spaces where the number of features is greater than the number of samples.
- **Non-linear Separation**: Use the "kernel trick" to handle complex decision boundaries.
- **Robust Regression**: Predict continuous values while only being sensitive to errors within a "tube" (SVR).
- **Novelty Detection**: Identify if new data comes from the same distribution as training data (OneClassSVM).

## Key Algorithms
1. **`SVC` / `SVR`**:
   - The main models for classification and regression.
   - Use `kernel='rbf'` (default), `'poly'`, or `'linear'`.
2. **`LinearSVC` / `LinearSVR`**:
   - Faster than `SVC(kernel='linear')` for large datasets.
   - Does not support the kernel trick.
3. **`OneClassSVM`**:
   - Unsupervised outlier detection.

## Key Hyperparameters
- **`C`**: Regularization parameter. 
  - Large `C`: Smaller margin, fits training data better (risk of overfitting).
  - Small `C`: Larger margin, simplifies the decision boundary (risk of underfitting).
- **`gamma`**: Defines how far the influence of a single training example reaches.
  - Large `gamma`: Local influence (complex boundaries).
  - Small `gamma`: Global influence (simpler boundaries).

## Computational Complexity
- $O(n_f \cdot n_s^2)$ to $O(n_f \cdot n_s^3)$ for most solvers. Not suitable for datasets with $> 100,000$ samples.

## Code Snippet: SVC with RBF Kernel

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 1. Tuning C and Gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# 2. Linear SVC for large-scale data
from sklearn.svm import LinearSVC
l_svc = LinearSVC(C=1.0, max_iter=10000)
l_svc.fit(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Text Feature Extraction

The `feature_extraction.text` module allows you to convert text documents into numeric feature vectors.

## What can be done?
- **Tokenization**: Breaking strings into words or n-grams.
- **Weighting**: Calculate how important a word is to a document (TF-IDF).
- **Dimensionality Reduction**: Map large vocabularies to a fixed-size space (Hashing).

## Key Tools
1. **`CountVectorizer`**:
   - Creates a "Bag of Words" representation.
   - Counts the occurrences of each word in each document.
2. **`TfidfVectorizer`**:
   - (Term Frequency-Inverse Document Frequency).
   - Down-weights common words (like "the", "a") and up-weights rare, informative words.
3. **`HashingVectorizer`**:
   - Stateless and fast.
   - Maps tokens to indices using a hash function.
   - **Advantage**: Low memory footprint and supports out-of-core learning.

## Key Parameters
- `stop_words`: List of words to ignore (e.g., `'english'`).
- `ngram_range`: Range of n-grams to extract (e.g., `(1, 2)` for unigrams and bigrams).
- `max_df` / `min_df`: Filter out words that appear too frequently or too rarely.

## Code Snippet: TF-IDF Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Pipeline for Text Classification
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', MultinomialNB()),
])

text_clf.fit(docs_train, y_train)

# 2. Inspecting vocabulary
tfidf = text_clf.named_steps['tfidf']
vocab = tfidf.vocabulary_ # Dict mapping words to indices
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Scikit-learn Cheatsheet: Decision Trees

Decision Trees are non-parametric supervised learning methods used for classification and regression.

## What can be done?
- **Automatic Interaction Discovery**: Naturally capture non-linear relationships between features.
- **Interpretability**: Easily visualize exactly how a decision is made.
- **Handling Mixed Data**: Can handle both numerical and categorical data (though scikit-learn's current implementation requires encoding).

## Key Algorithms
1. **`DecisionTreeClassifier` / `DecisionTreeRegressor`**:
   - Predictive models that split data into branches by optimizing info gain (Gini/Entropy).
2. **`ExtraTrees`**:
   - Randomly picks splits to further reduce variance (often used in ensembles).

## Important Concepts
- **Overfitting**: Trees can grow very deep and learn the noise in data.
- **Pruning (`ccp_alpha`)**: Cost Complexity Pruning is used to remove branches that provide little predictive power.
- **Feature Importance**: Trees provide a ranking of features based on how much they reduce the impurity.

## Visualization
- **`plot_tree`**: Render the tree structure as a diagram.
- **`export_text`**: Get a text-based summary of rules.

## Code Snippet: Tree Building & Visualization

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

# 1. Fit Tree
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf.fit(X, y)

# 2. Visualize
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# 3. Export Rules
tree_rules = export_text(clf, feature_names=list(feature_names))
print(tree_rules)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.


---


# Technical Writing Cheatsheet

Code is read by machines; documentation is read by humans. Both require clarity, structure, and optimization.
Based on *Google Developer Documentation Style Guide*, *The Elements of Style*, and *理科系の作文技術 (Technical Writing for Science and Engineering)*.

## 1. Core Principles (The "Linting" Rules)

### Active Voice
Make it clear *who* is performing the action. Passive voice obscures responsibility and bloats text.

*   **Bad**: The data is processed by the function.
*   **Good**: The function processes the data.
*   **Bad**: It is recommended to use JAX for acceleration.
*   **Good**: We recommend using JAX for acceleration. / Use JAX for acceleration.

### Second Person (Use "You")
Address the user directly. Third-person terms like `User` or `Developer` create unnecessary distance.

*   **Bad**: The user needs to install dependencies first.
*   **Good**: Install dependencies first. / You must install dependencies.

### Present Tense
Documentation describes what exists *now*. Avoid future tense (`will`) unless promising a future feature.

*   **Bad**: The script will run the simulation.
*   **Good**: The script runs the simulation.
*   **Bad**: Running this command will cause an error.
*   **Good**: Running this command causes an error.

---

## 2. Structure & Layout (Refactoring Text)

### BLUF (Bottom Line Up Front)
State the conclusion first. Readers are busy; do not bury the lead.

*   **Paragraph**: The first sentence (Topic Sentence) should summarize the paragraph.
*   **Document**: The first section should explain "What is this?" and "Why does it matter?".

### Lists over Prose
Break down complex series (3+ items) into lists.

**Bad**:
To optimize the model, you can adjust the learning rate, change the batch size, and increase the number of epochs.

**Good**:
To optimize the model:
*   Adjust the learning rate.
*   Change the batch size.
*   Increase the number of epochs.

### Parallelism in Lists
Ensure all list items start with the same part of speech (usually imperative verbs).

*   **Bad**:
    *   Download the file.
    *   Configuration of the server.
    *   To start the app, run `main.py`.
*   **Good**:
    *   Download the file.
    *   Configure the server.
    *   Start the app by running `main.py`.

---

## 3. Clarity & Conciseness (Optimization)

### Kill Filler Words
Delete words that add no meaning. Treat them like compiler warnings.

| Remove | Why |
| :--- | :--- |
| **basically** / **essentially** | Adds no information. |
| **very** / **really** / **quite** | Weakens adjectives. Use stronger words or specific numbers. |
| **in order to** | "to" is sufficient. |
| **at this point in time** | "now" or "currently" is sufficient. |
| **needless to say** | If it's needless, don't say it. |

### Short Sentences
One idea per sentence. Long sentences breed ambiguity. Break them up.

*   **Bad**: The API is fast, but it is experimental, so you should use it with caution because it might change.
*   **Good**: The API is fast but experimental. Use it with caution; it might change.

### Clarify Antecedents
Avoid vague `it`, `this`, `that`.

*   **Bad**: Python is slow. JAX compiles it to XLA. **This** makes it faster.
*   **Good**: Python is slow. JAX compiles Python to XLA. **This compilation** makes execution faster.

---

## 4. Specific Artifacts Templates

### README.md Structure
The landing page. Answer "What is this?" in 3 seconds.

1.  **Project Name & One-line Description**: What is this?
2.  **Badges**: CI status, License, Version.
3.  **Why this?**: Differentiators, value proposition.
4.  **Quick Start**: Minimal installation and "Hello World" code.
5.  **Installation**: Detailed setup instructions.
6.  **Usage/Examples**: Common use cases.

### Git Commit Messages (Conventional Commits)
Machine-parseable history.

Format: `<type>(<scope>): <subject>`

*   `feat`: New feature
*   `fix`: Bug fix
*   `docs`: Documentation only
*   `style`: Formatting (whitespace, semi-colons, etc.)
*   `refactor`: Code change that neither fixes a bug nor adds a feature
*   `perf`: A code change that improves performance
*   `test`: Adding missing tests or correcting existing tests
*   `chore`: Build process or auxiliary tool changes

**Example**:
`feat(solver): implement sparse matrix support for GMRES`

---

## 5. Tools (Linter for Prose)

Just as code has linters, prose has linters.

*   **[Vale](https://vale.sh/)**: CLI-based prose linter. Enforces style guides (Google, Microsoft) in CI.
*   **[Hemingway Editor](https://hemingwayapp.com/)**: Visualizes sentence complexity. Highlights passive voice and adverbs.
*   **[Grammarly](https://www.grammarly.com/)**: Standard grammar and spell checker.

---

## 6. Common Typos & Gotchas in Tech

*   **Setup** (noun) vs **Set up** (verb)
    *   "Check the **setup**." / "Please **set up** the env."
*   **Login** (noun/adj) vs **Log in** (verb)
    *   "Login page" / "Please log in."
*   **i.e.** (that is) vs **e.g.** (for example)
    *   Often confused. If unsure, use English words ("that is", "for example").
*   **data**
    *   Technically plural, but accepted as singular (uncountable) in modern usage. Be consistent.
    *   "The data **is** processed." (Preferred)

---

## 7. 理科系の作文技術

木下是雄著『理科系の作文技術』に基づく。

### 7.1. 基本姿勢

#### 事実と意見の区別
最も重要な原則。「事実（Fact）」と「意見（Opinion）」を厳密に区別する。

*   **事実**: 証拠を示してその真偽を客観的に確認できる記述。自然現象、実験結果、数式など。
    *   *例*: 「AとBを混合すると爆発した。」
*   **意見**: 推論、判断、感想、仮説。
    *   *例*: 「AとBの混合は**危険である**。」（"危険"は判断）
*   **ルール**: 事実の記述の中に、主観的な形容詞（美しい、悲惨な、画期的な）を混ぜない。「〜であると思う」と「〜である」を使い分ける。

#### 読者の設定
「誰に読ませるか」を最初に決める。
*   特定の専門家向けか？ 一般向けか？
*   読者が持っている前提知識は何か？
*   **ルール**: 読者が知りたい情報を、読者が理解できる言葉で書く。自己満足の日記ではない。

### 7.2. 構成

#### 起承転結の否定
理系の文章に「起承転結」は不要であり、有害ですらある。
*   **文学**: クライマックス（転・結）まで結論を隠す。
*   **理系**: **結論が先**（Conclusion First）。

#### 重点先行主義
1.  **目標規定文**: この文書は何を主張するものか、最初に一文で定義する。
2.  **要約（Summary）**: 結論と重要ポイントを最初に書く。
3.  **本論**: 詳細な論証。
4.  **結論**: まとめ。

### 7.3. パラグラフ

*   **一トピック一パラグラフ**: 1つのパラグラフには1つのトピックだけを入れる。
*   **トピックセンテンス**: パラグラフの最初の文（トピックセンテンス）で、その段落の要約や主張を述べる。残りの文はその展開や裏付け。
*   **接続詞**: 接続詞（しかし、したがって）を使って、パラグラフ間の論理的なつながりを明確にする。

### 7.4. 文体

#### 明確な主語と述語
日本語は主語を省略しがちだが、技術文では誤解を招く。
*   **Bad**: データを入力するとエラーになった。（何が？誰が？）
*   **Good**: ユーザーがデータを入力すると、システムはエラーを返した。

#### 簡潔な表現
*   **逆茂木（さかもぎ）文を避ける**: 修飾語が長すぎて主語になかなかたどり着かない文。
    *   *Bad*: 非常に高速で、メモリ効率も良く、最新のGPU向けに最適化された**ライブラリ**。
    *   *Good*: この**ライブラリ**は高速かつ省メモリで、最新のGPUに最適化されている。
*   **無意味な繋ぎ言葉を削る**: 「〜における」「〜に関する」「〜を行う」などは多くの場合削除できる。
    *   *Bad*: データの解析**を行う**。
    *   *Good*: データを解析する。
    *   *Bad*: パフォーマンス**に関する**問題。
    *   *Good*: パフォーマンスの問題。

#### 漢字と平仮名のバランス
*   難しい漢字を使いすぎない。
*   接続詞や副詞は平仮名にする（例：「〜の**ため**」「**かつ**」「**また**」）。
*   「書けるけれど書かない」勇気を持つ。


---


# uv Cheat Sheet

uv is an extremely fast Python package and project manager, written in Rust. It serves as a unified replacement for `pip`, `pip-tools`, `poetry`, `pyenv`, `virtualenv`, and more.

## Installation and Setup

**Install uv:**
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip (not recommended for main usage, but possible)
pip install uv
```

**Upgrade uv:**
```bash
uv self update
```

**Shell Autocompletion:**
```bash
# Example for bash
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

---

## Project Management
uv allows you to manage Python projects with `pyproject.toml` and a cross-platform `uv.lock`.

**Initialize a Project:**
```bash
# Create a new project
uv init my-project

# Initialize in current directory
uv init
```

**Manage Dependencies:**
```bash
# Add a dependency
uv add requests

# Add a specific version
uv add 'requests==2.31.0'

# Add a development dependency
uv add --dev pytest

# Remove a dependency
uv remove requests
```

**Sync Environment:**
```bash
# Sync virtual environment with lockfile (creates .venv if needed)
uv sync

# specific extras
uv sync --all-extras
```

**Run Commands in Project Scope:**
```bash
# Runs command in the project's environment (auto-syncs if needed)
uv run python main.py
uv run pytest
```

---

## Scripts
uv can run standalone scripts and manage their dependencies automatically without manual virtual environments.

**Run a Script:**
```bash
# Run a script (auto-creates ephemeral environment)
uv run script.py
```

**Run with Dependencies (Ad-hoc):**
```bash
# Requires 'rich' and 'requests' available for this run
uv run --with rich --with requests script.py
```

**Inline Script Metadata:**
Standardized way to declare dependencies inside the script (PEP 723).
```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint
# ...
```
Run it simply with:
```bash
uv run script.py
```

---

## Tool Management
Execute and install Python command-line tools (similar to `pipx`).

**Run a Tool Ephemerally (`uvx`):**
```bash
# Run 'ruff' without installing it globally
uvx ruff check .

# Run a secure shell with 'httpie'
uvx httpie https://google.com
```

**Install Tools Globally:**
```bash
# Install 'ruff' tool
uv tool install ruff

# List installed tools
uv tool list

# Upgrade installed tools
uv tool upgrade --all
```

---

## Python Version Management
uv can manage Python installations itself, replacing `pyenv`.

**Install Python:**
```bash
# Install latest Python
uv python install

# Install specific versions
uv python install 3.12 3.11
```

**List Available Versions:**
```bash
uv python list
```

**Pin Python Version for Project:**
```bash
# Creates/Updates .python-version file
uv python pin 3.11
```

---

## Pip Interface
uv provides a drop-in replacement for `pip` commands, useful for legacy workflows or CI.

```bash
# Install packages into current environment
uv pip install requests

# specific requirements file
uv pip install -r requirements.txt

# Compile requirements (pip-compile replacement)
uv pip compile requirements.in -o requirements.txt

# Sync environment (pip-sync replacement)
uv pip sync requirements.txt
```
---

## Configuration & Environment

**Common Environment Variables:**
- `UV_CACHE_DIR`: Directory for uv cache.
- `UV_PROJECT_ENVIRONMENT`: Path to the virtual environment (default: `.venv`).
- `UV_PYTHON_DOWNLOADS`: Control python downloads (`auto` [default], `manual`, `never`).
- `UV_NATIVE_TLS`: Set `true` to use system certificate store (important for corporate certs).
- `HTTP_PROXY` / `HTTPS_PROXY`: Standard proxy settings.
- `UV_INDEX_URL` / `UV_EXTRA_INDEX_URL`: Default and extra package indexes.

**pyproject.toml Configuration:**
Example for a data analysis project (not installed as a package).

```toml
[project]
name = "analysis-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas",
    "numpy",
    "jupyterlab",
    "polars",
    # GitHub dependency (name only here, details in tool.uv.sources)
    "internal-tools", 
]

[tool.uv]
package = false # Important: Prevents building this project itself as a package

[tool.uv.sources]
# robust reproducibility: pin git tag/commit for github deps
internal-tools = { git = "https://github.com/my-org/internal-tools", tag = "v1.0.2" }
torch = { index = "pytorch" }

# Custom Index (e.g. for CUDA specific pytorch)
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
```

---

## Single-File Scripts & Metadata
PEP 723 allows defining dependencies directly within a script file. This replaces the need for `uv run --no-project --with ...` in many persistent cases.

**Example `script.py`:**
```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests<3",
#     "rich",
# ]
# ///

import requests
from rich.pretty import pprint

print("Running with self-contained dependencies!")
```

**Run it:**
```bash
uv run script.py
```
uv will automatically detect the metadata block, create an ephemeral environment, install `requests` and `rich`, and execute the script.

---

## Useful Flags & Concepts

- **`--no-project`**: Run a command ignoring the current project context.
- **`--with`**: Add ephemeral dependencies to a run command.
- **`--refresh`**: Force refresh of cached data.
- **Cache**: uv caches aggressively. Use `uv cache clean` to clear.


---


