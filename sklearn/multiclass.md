# Scikit-learn Cheatsheet: Multiclass

Multiclass classification refers to those classification tasks that have more than two targets.

## What can be done?
- **Convert Binary to Multiclass**: Use meta-estimators to apply binary classifiers (like SVM) to multiclass problems.
- **Decision Strategy**: Choose between "One-vs-Rest" or "One-vs-One".

## Key Algorithms
1. **`OneVsRestClassifier` (OvR)**:
   - Fits one classifier per class.
   - Computational burden is $N$ classifiers.
   - Most common and scalable strategy.
2. **`OneVsOneClassifier` (OvO)**:
   - Fits one classifier for every pair of classes.
   - Computational burden is $N \cdot (N-1) / 2$ classifiers.
   - Often used for algorithms that don't scale well with the volume of data (like kernels).
3. **`OutputCodeClassifier`**:
   - Uses Error-Correcting Output Codes. Each class is represented by a binary code.

## Native Multiclass support
Some estimators handle multiclass natively (no wrapper needed):
- `LogisticRegression` (using `'multinomial'` loss).
- `RandomForestClassifier`.
- `GaussianNB`.
- `LinearDiscriminantAnalysis`.

## Code Snippet: Multiclass Strategies

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# 1. One-vs-Rest
ovr = OneVsRestClassifier(SVC(kernel='linear'))
ovr.fit(X, y)

# 2. One-vs-One
ovo = OneVsOneClassifier(SVC(kernel='linear'))
ovo.fit(X, y)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
