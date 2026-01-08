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

