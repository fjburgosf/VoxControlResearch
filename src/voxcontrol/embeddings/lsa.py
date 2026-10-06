"""Local sentence embedding: character + word TF-IDF projected with truncated SVD (LSA).

Fully offline and deterministic given the seed. Character n-grams make it
tolerant to ASR spelling errors. A neural sentence encoder can be plugged in
through the same ``fit`` / ``transform`` interface.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import normalize


def make_tfidf_pair() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                           preprocessor=normalize, min_df=1)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                           preprocessor=normalize, token_pattern=r"(?u)\b\w+\b", min_df=1)
    return char, word


class SparseTfidf:
    """Concatenated character and word TF-IDF features."""

    def __init__(self):
        self.char, self.word = make_tfidf_pair()

    def fit(self, texts: list[str]) -> "SparseTfidf":
        self.char.fit(texts)
        self.word.fit(texts)
        return self

    def transform(self, texts: list[str]):
        return hstack([self.char.transform(texts), self.word.transform(texts)]).tocsr()

    def fit_transform(self, texts: list[str]):
        return self.fit(texts).transform(texts)

    @property
    def vocabulary(self) -> set[str]:
        return set(self.word.vocabulary_)


class LSAEmbedder:
    def __init__(self, n_components: int = 192, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.tfidf = SparseTfidf()
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        X = self.tfidf.fit_transform(texts)
        k = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(X)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        Z = self.svd.transform(self.tfidf.transform(texts)).astype(np.float64)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(norms, 1e-12)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
