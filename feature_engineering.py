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
    X = vectorizer.fit_transform(data["processed_text"])
    df_count= pd.DataFrame(
        X.toarray(),
        columns = [f"count_{s}" for s in vectorizer.get_feature_names_out()]
    )
    if features.lower() == "count":
        return df_count
    XX = tfidf.fit_transform(X)
    df_tfidf = pd.DataFrame(
        XX.toarray(),
        columns = [f"tfidf_{s}" for s in vectorizer.get_feature_names_out()]
    )
    if features.lower() == "tfidf":
        return df_tfidf
    elif features.lower() == "both":
        return pd.concat([df_count, df_tfidf], axis = 1)
    else:
        raise RuntimeError("Invalid features. features can only take values 'tfidf' or 'count'.")
