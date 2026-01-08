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
