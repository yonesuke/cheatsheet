# Scikit-learn Cheatsheet: Specialized Applications

This cheatsheet covers real-world application examples and specialized techniques demonstrated in the scikit-learn gallery.

## Domain-Specific Applications

### 1. Computer Vision & Face Recognition
- **Technique**: Eigenfaces (PCA) + Classification (SVM).
- **Goal**: Identify individuals from images.
- **Key Function**: `sklearn.decomposition.PCA`, `sklearn.svm.SVC`.

### 2. Finance & Ecology
- **Stock Market Embedding**: Discovering structure in the stock market using `GraphicalLassoCV` to learn the covariance matrix of price fluctuations.
- **Species Distribution**: Modeling the geographic distribution of species using `OneClassSVM` or `RandomForest`.

### 3. Text Mining & Topic Modeling
- **Topics Extraction**: Uncovering latent themes in document collections.
- **Key Function**: `sklearn.decomposition.NMF`, `sklearn.decomposition.LatentDirichletAllocation`.

## Advanced Engineering Techniques

### 4. Out-of-Core Classification
- **Scenario**: Dataset too large to fit in memory (Incremental Learning).
- **Key Function**: `partial_fit()` method available in models like `SGDClassifier`, `MultinomialNB`, and `HashingVectorizer`.

### 5. Time Series Forecasting
- **Technique**: Lagged features and cyclical engineering (sine/cosine transforms).
- **Goal**: Convert time-based data into features suitable for regression.

### 6. Outlier & Novelty Detection
- **Technique**: Identifying "weird" samples in a dataset.
- **Key Function**: `sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `sklearn.svm.OneClassSVM`.

## Computational Considerations
- **Prediction Latency**: Using `plot_prediction_latency` to analyze how different models scale with the number of features.
- **Model Complexity**: Analyzing the trade-off between model complexity (e.g., number of trees) and accuracy.

## Code Snippet: Out-of-Core Learning

```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

# HashingVectorizer is stateless and memory-efficient
vectorizer = HashingVectorizer(stop_words='english')
clf = SGDClassifier(loss='log_loss')

# Simulated stream of data
for X_batch_raw, y_batch in stream_generator():
    X_batch = vectorizer.transform(X_batch_raw)
    # partial_fit allows learning from data bit by bit
    clf.partial_fit(X_batch, y_batch, classes=all_classes)
```

## Code Snippet: Cyclical Feature Engineering (for Time Series)

```python
import numpy as np
import pandas as pd

# Convert hour of day (0-23) into cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
