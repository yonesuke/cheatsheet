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
