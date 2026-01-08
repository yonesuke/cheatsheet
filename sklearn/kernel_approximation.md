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
