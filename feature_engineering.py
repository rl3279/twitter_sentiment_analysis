import numpy as np
import pandas as pd
from preprocessing import preprocess_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def exclaim_freq(s: str) -> float:
    """Frequency of excalamtion points in a tweet.

    :param s: input str
    :type s: str
    :rtype: float
    """
    s = "".join(s.split())
    count = sum([1 if t == "!" else 0 for t in s])
    return 0 if len(s) == 0 else count / len(s) 


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
    s = preprocess_pipeline(s, return_lower=False)
    count = sum([1 if t.isupper() else 0 for t in s])
    return 0 if len(s) == 0 else count / len(s) 

def get_token_features(data: pd.Series, features = "tfidf") -> pd.DataFrame:
    """Encode a Series of text string to TF-IDF.

    :param data: input data
    :type data: pd.Series
    :rtype: np.ndarray
    """
    vectorizer = CountVectorizer()
    tfidf = TfidfTransformer()
    X = vectorizer.fit_transform(data)
    if features.lower() == "count":
        df = pd.DataFrame(
            X.toarray(),
            columns = [f"count_{s}" for s in vectorizer.get_feature_names_out()]
        )
    elif features.lower() == "tfidf":
        X = tfidf.fit_transform(X)
        df = pd.DataFrame(
            X.toarray(),
            columns = [f"tfidf_{s}" for s in vectorizer.get_feature_names_out()]
        )
    else:
        raise RuntimeError("Invalid features. features can only take values 'tfidf' or 'count'.")
    return df
