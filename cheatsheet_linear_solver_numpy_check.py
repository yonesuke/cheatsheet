# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
# ]
# ///

import numpy as np
from numpy.typing import NDArray
import dataclasses
from typing import Tuple

@dataclasses.dataclass
class SolverResult:
    solution: NDArray[np.float64]
    name: str
    error_norm: float
    converged: bool = True

def lu_decomposition_simple(A: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes LU decomposition (A = LU) without pivoting.
    
    Args:
        A: Square matrix of shape (N, N).
        
    Returns:
        Tuple (L, U) where L is lower triangular with unit diagonal and U is upper triangular.
    """
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = A.copy()
    
    for k in range(n - 1):
        for i in range(k + 1, n):
            if np.isclose(U[k, k], 0.0):
                raise ValueError("Zero pivot encountered. This simplified LU requires non-singular, well-conditioned matrices.")
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]
            
    return L, U

def solve_lu(L: NDArray[np.float64], U: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solves Ax = b given A = LU."""
    n = L.shape[0]
    # Forward substitution Ly = b
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
    # Backward substitution Ux = y
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
    return x

def cholesky_decomposition(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes Cholesky decomposition A = LL^T.
    
    Args:
        A: Symmetric Positive Definite matrix.
        
    Returns:
        Lower triangular matrix L.
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            sum_val = np.sum(L[i, :j] * L[j, :j])
            
            if i == j:
                val = A[i, i] - sum_val
                if val <= 0:
                    raise ValueError("Matrix is not positive definite.")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (1.0 / L[j, j]) * (A[i, j] - sum_val)
                
    return L

def solve_cholesky(L: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solves Ax = b given A = LL^T."""
    # Solve Ly = b
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        
    # Solve L^T x = y
    LT = L.T
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i+1:], x[i+1:])) / LT[i, i]
        
    return x

def qr_householder(A: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes QR decomposition using Householder reflections.
    
    Returns:
        Tuple (Q, R) where Q is orthogonal and R is upper triangular.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        # Determine sign to avoid cancellation
        alpha = -np.sign(x[0]) if x[0] != 0 else 1.0
        norm_x = np.linalg.norm(x)
        
        u = x - alpha * norm_x * e1
        v = u / np.linalg.norm(u)
        
        # Apply transformation to R
        R[k:, k:] -= 2.0 * np.outer(v, np.dot(v, R[k:, k:]))
        
        # Apply transformation to Q
        Q[k:, :] -= 2.0 * np.outer(v, np.dot(v, Q[k:, :]))
        
    return Q.T, R

def solve_qr(Q: NDArray[np.float64], R: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solves Ax = b given A = QR => Rx = Q^T b."""
    # y = Q^T b
    y = np.dot(Q.T, b)
    
    # Solve Rx = y (Backward substitution)
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
        
    return x

def tdma_solver(lower: NDArray[np.float64], diag: NDArray[np.float64], upper: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Thomas Algorithm (TDMA) for tridiagonal systems.
    
    Args:
        lower: Sub-diagonal (length N-1 or N with first 0).
        diag: Main diagonal (length N).
        upper: Super-diagonal (length N-1 or N with last 0).
        d: Constant vector (length N).
    """
    n = len(d)
    # Adjust inputs if length N passed for off-diagonals (common in some APIs)
    l_calc = lower[1:] if len(lower) == n else lower
    u_calc = upper[:-1] if len(upper) == n else upper
    
    # Copies to avoid modifying inputs
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    
    # Forward elimination
    c_prime[0] = u_calc[0] / diag[0]
    d_prime[0] = d[0] / diag[0]
    
    for i in range(1, n - 1):
        temp = diag[i] - l_calc[i-1] * c_prime[i-1]
        c_prime[i] = u_calc[i] / temp
        d_prime[i] = (d[i] - l_calc[i-1] * d_prime[i-1]) / temp
        
    # Last step for d_prime
    i = n - 1
    temp = diag[i] - l_calc[i-1] * c_prime[i-1]
    d_prime[i] = (d[i] - l_calc[i-1] * d_prime[i-1]) / temp
    
    # Backward substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

def conjugate_gradient(A: NDArray[np.float64], b: NDArray[np.float64], x0: NDArray[np.float64] = None, tol: float = 1e-8, max_iter: int = 1000) -> NDArray[np.float64]:
    """Conjugate Gradient method for SPD matrices."""
    if x0 is None:
        x0 = np.zeros_like(b)
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rho = np.dot(r, r)
    
    for _ in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        if pAp == 0: break
        
        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap
        
        new_rho = np.dot(r, r)
        if np.sqrt(new_rho) < tol:
            break
            
        beta = new_rho / rho
        p = r + beta * p
        rho = new_rho
        
    return x

def gmres(A: NDArray[np.float64], b: NDArray[np.float64], x0: NDArray[np.float64] = None, m: int = 30, tol: float = 1e-8, max_restarts: int = 10) -> NDArray[np.float64]:
    """
    Restarted GMRES(m).
    
    Args:
        m: Restart parameter (Krylov subspace dimension).
    """
    if x0 is None:
        x0 = np.zeros_like(b)
    x = x0.copy()
    n = len(b)
    
    for _ in range(max_restarts):
        r0 = b - A @ x
        beta = np.linalg.norm(r0)
        if beta < tol:
            return x
        
        V = [r0 / beta] # Basis vectors
        H = np.zeros((m + 1, m)) # Hessenberg matrix
        
        k = 0
        for j in range(m):
            k = j
            w = A @ V[j]
            
            # Arnoldi Process (Gram-Schmidt)
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[i])
                w = w - H[i, j] * V[i]
            
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-12: # Check for breakdown
                break
            V.append(w / H[j + 1, j])
            
            # Solve minimized least squares H_k * y = beta * e1
            # Using numpy.linalg.lstsq for the small system solution
            e1 = np.zeros(j + 2)
            e1[0] = beta
            
            y, _, _, _ = np.linalg.lstsq(H[:j+2, :j+1], e1, rcond=None)
            
            curr_error = np.linalg.norm(H[:j+2, :j+1] @ y - e1)
            if curr_error < tol:
                break

        # Reconstruct x
        # x = x0 + V_k @ y
        # V is list of vectors, stack column-wise then take first k+1 columns (since j loop could verify early)
        V_mat = np.column_stack(V[:-1]) # remove last vector which is V_{k+1}
        
        # Re-solve y for final k state just to be safe/consistent
        e1 = np.zeros(k + 2)
        e1[0] = beta
        y_final, _, _, _ = np.linalg.lstsq(H[:k+2, :k+1], e1, rcond=None)
        
        x = x + V_mat[:, :k+1] @ y_final
        
        if np.linalg.norm(b - A @ x) < tol:
            break
            
    return x

def bicgstab(A: NDArray[np.float64], b: NDArray[np.float64], x0: NDArray[np.float64] = None, tol: float = 1e-8, max_iter: int = 1000) -> NDArray[np.float64]:
    """Bi-Conjugate Gradient Stabilized method."""
    if x0 is None:
        x0 = np.zeros_like(b)
    x = x0.copy()
    r = b - A @ x
    r_hat = r.copy()
    p = r.copy()
    rho = np.dot(r_hat, r)
    
    for _ in range(max_iter):
        v = A @ p
        alpha_denom = np.dot(r_hat, v)
        if alpha_denom == 0: break
        
        alpha = rho / alpha_denom
        s = r - alpha * v
        if np.linalg.norm(s) < tol:
            x += alpha * p
            break
            
        t = A @ s
        omega_denom = np.dot(t, t)
        if omega_denom == 0: break
        omega = np.dot(t, s) / omega_denom
        
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        if np.linalg.norm(r) < tol:
            break
            
        new_rho = np.dot(r_hat, r)
        if new_rho == 0: break
        
        beta = (new_rho / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        rho = new_rho
        
    return x

def run_verifications():
    print(f"{'Algorithm':<20} | {'Status':<10} | {'Error Norm':<15}")
    print("-" * 50)
    
    np.random.seed(42)
    N = 50
    
    # 1. Check LU (Need non-singular, well-conditioned. Diagonally dominant is safe)
    A_lu = np.random.randn(N, N)
    A_lu += np.eye(N) * N * 2
    b_lu = np.random.randn(N)
    
    L, U = lu_decomposition_simple(A_lu)
    x_lu = solve_lu(L, U, b_lu)
    check_lu = np.linalg.norm(A_lu @ x_lu - b_lu)
    print(f"{'LU (No Pivot)':<20} | {'PASS' if check_lu < 1e-8 else 'FAIL':<10} | {check_lu:.2e}")
    
    # 2. Check Cholesky (Requires SPD)
    A_tmp = np.random.randn(N, N)
    A_chol = A_tmp @ A_tmp.T + np.eye(N) # Ensure Positive Definite
    b_chol = np.random.randn(N)
    
    L_chol = cholesky_decomposition(A_chol)
    x_chol = solve_cholesky(L_chol, b_chol)
    check_chol = np.linalg.norm(A_chol @ x_chol - b_chol)
    print(f"{'Cholesky':<20} | {'PASS' if check_chol < 1e-8 else 'FAIL':<10} | {check_chol:.2e}")
    
    # 3. Check QR
    A_qr = np.random.randn(N, N)
    b_qr = np.random.randn(N)
    
    Q, R = qr_householder(A_qr)
    x_qr = solve_qr(Q, R, b_qr)
    check_qr = np.linalg.norm(A_qr @ x_qr - b_qr)
    print(f"{'QR':<20} | {'PASS' if check_qr < 1e-8 else 'FAIL':<10} | {check_qr:.2e}")
    
    # 4. Check TDMA
    # Create tridiagonal system
    lower = np.random.rand(N-1)
    upper = np.random.rand(N-1)
    diag = np.random.rand(N) + 2 # Diagonally dominant
    d_tdma = np.random.rand(N)
    
    # Construct density matrix for verification
    A_tdma = np.diag(diag) + np.diag(upper, k=1) + np.diag(lower, k=-1)
    
    x_tdma = tdma_solver(lower, diag, upper, d_tdma)
    check_tdma = np.linalg.norm(A_tdma @ x_tdma - d_tdma)
    print(f"{'TDMA':<20} | {'PASS' if check_tdma < 1e-8 else 'FAIL':<10} | {check_tdma:.2e}")
    
    # 5. Check CG (SPD required)
    x_cg = conjugate_gradient(A_chol, b_chol)
    check_cg = np.linalg.norm(A_chol @ x_cg - b_chol)
    print(f"{'CG':<20} | {'PASS' if check_cg < 1e-5 else 'FAIL':<10} | {check_cg:.2e}")
    
    # 6. Check GMRES (General Matrix)
    x_gmres = gmres(A_lu, b_lu, m=20)
    check_gmres = np.linalg.norm(A_lu @ x_gmres - b_lu)
    print(f"{'GMRES(20)':<20} | {'PASS' if check_gmres < 1e-5 else 'FAIL':<10} | {check_gmres:.2e}")
    
    # 7. Check BiCGStab (General Matrix)
    x_bicg = bicgstab(A_lu, b_lu)
    check_bicg = np.linalg.norm(A_lu @ x_bicg - b_lu)
    print(f"{'BiCGStab':<20} | {'PASS' if check_bicg < 1e-5 else 'FAIL':<10} | {check_bicg:.2e}")

if __name__ == "__main__":
    run_verifications()
