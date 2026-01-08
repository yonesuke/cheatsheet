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
