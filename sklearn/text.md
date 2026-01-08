# Scikit-learn Cheatsheet: Text Feature Extraction

The `feature_extraction.text` module allows you to convert text documents into numeric feature vectors.

## What can be done?
- **Tokenization**: Breaking strings into words or n-grams.
- **Weighting**: Calculate how important a word is to a document (TF-IDF).
- **Dimensionality Reduction**: Map large vocabularies to a fixed-size space (Hashing).

## Key Tools
1. **`CountVectorizer`**:
   - Creates a "Bag of Words" representation.
   - Counts the occurrences of each word in each document.
2. **`TfidfVectorizer`**:
   - (Term Frequency-Inverse Document Frequency).
   - Down-weights common words (like "the", "a") and up-weights rare, informative words.
3. **`HashingVectorizer`**:
   - Stateless and fast.
   - Maps tokens to indices using a hash function.
   - **Advantage**: Low memory footprint and supports out-of-core learning.

## Key Parameters
- `stop_words`: List of words to ignore (e.g., `'english'`).
- `ngram_range`: Range of n-grams to extract (e.g., `(1, 2)` for unigrams and bigrams).
- `max_df` / `min_df`: Filter out words that appear too frequently or too rarely.

## Code Snippet: TF-IDF Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Pipeline for Text Classification
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', MultinomialNB()),
])

text_clf.fit(docs_train, y_train)

# 2. Inspecting vocabulary
tfidf = text_clf.named_steps['tfidf']
vocab = tfidf.vocabulary_ # Dict mapping words to indices
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
