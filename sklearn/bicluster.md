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
