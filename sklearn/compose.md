# Scikit-learn Cheatsheet: Compose (Pipelines & Meta-Estimators)

The `compose` module provides tools to combine multiple estimators into a single one, facilitating complex workflows and preventing data leakage.

## What can be done?
- **Chaining Steps**: Sequentially apply transformers and a final estimator.
- **Heterogeneous Data**: Apply different transformations to different columns (e.g., one-hot encoding for categorical, scaling for numerical).
- **Parallel Processing**: Combine multiple feature extraction methods.
- **Target Transformation**: Transform the target variable (e.g., log transform) automatically during fit/predict.

## Key Tools
1. **`Pipeline`**:
   - Chains multiple steps. Only the last step can be an estimator (model), others must be transformers.
   - Ensures that transformers are fitted on training data and applied to test data correctly.
2. **`ColumnTransformer`**:
   - Routes specific columns to specific transformers.
   - Essential for handling mixed-type data (numeric + categorical).
3. **`FeatureUnion`**:
   - Concatenates the results of multiple transformer objects.
4. **`TransformedTargetRegressor`**:
   - Wraps a regressor to apply a transformation to the target $y$ before fitting and an inverse transformation after predicting.

## Best Practices
- **Prevent Data Leakage**: Always use Pipelines when performing cross-validation to ensure preprocessing parameters (like mean/std) are calculated only on the training folds.
- **Hyperparameter Tuning**: You can tune hyperparameters of any step in a pipeline using `stepname__parametername` syntax in `GridSearchCV`.

## Code Snippet: Pipeline & ColumnTransformer

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 1. Define transformers for different column types
numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_features = ["embarked", "sex"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# 2. Bundle them in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 3. Create the final Pipeline
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

# 4. Use it as a single estimator
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
