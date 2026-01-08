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
