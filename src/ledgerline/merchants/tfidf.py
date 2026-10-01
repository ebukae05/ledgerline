"""TF-IDF on character n-grams + logistic regression.

Character n-grams (2-5 letters) instead of words, because descriptors are
abbreviated and misspelled: "MKTPL", "MARKTPLACE" and "MARKETPLACE" share
most of their character chunks but no whole word.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42
C_GRID = (0.3, 1.0, 3.0, 10.0)
DIGIT = re.compile(r"\d")


def _prepare(text: str) -> str:
    # Keep the [debit]/[credit] sign; map every digit to 0 so store numbers
    # and reference codes look alike instead of becoming thousands of features.
    return DIGIT.sub("0", text.lower())


def build(c: float = 3.0) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=_prepare,
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("model", LogisticRegression(C=c, max_iter=3000, random_state=SEED)),
        ]
    )


def fit(descriptions: list[str], categories: list[str], c: float = 3.0) -> Pipeline:
    return build(c).fit(descriptions, categories)


def predict_with_confidence(
    model: Pipeline, descriptions: list[str]
) -> tuple[list[str], np.ndarray]:
    proba = model.predict_proba(descriptions)
    labels = model.classes_[proba.argmax(axis=1)]
    return labels.tolist(), proba.max(axis=1)
