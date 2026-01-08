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
