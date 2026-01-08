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
