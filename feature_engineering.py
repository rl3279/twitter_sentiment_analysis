import numpy as np
import pandas as pd
from preprocessing import preprocess_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def exlaim_freq(s: str) -> float:
    """Frequency of excalamtion points in a tweet.

    :param s: input str
    :type s: str
    :rtype: float
    """
    s = "".join(s.split())
    count = sum([1 if t == "!" else 0 for t in s])
    return count / len(s)


def mention_count(s: str) -> int:
    """Counts how many mentions occurred in a tweet.

    :param s: input str
    :type s: str
    :rtype: int
    """
    count = sum([1 if t.startswith("@") else 0 for t in s.split()])
    return count


def cap_freq(s: str) -> float:
    """Frequency of capitalized letter usage in a tweet.

    :param s: input str
    :type s: str
    :rtype: float
    """
    s = preprocess_pipeline(s)
    count = sum([1 if t.isupper() else 0 for t in s])
    return count / len(s)


def get_tfidf(data: pd.Series) -> np.ndarray:
    """Encode a Series of text string to TF-IDF.

    :param data: input data
    :type data: pd.Series
    :rtype: np.ndarray
    """
    vectorizer = CountVectorizer()
    tfidf = TfidfTransformer()
    X = vectorizer.fit_transform(data)
    X = tfidf.fit_transform(X)
    return X.toarray()
