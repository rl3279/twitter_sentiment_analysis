import numpy as np
import pandas as pd

from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec
from preprocessing import preprocess_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler


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


def get_token_features(data: pd.Series, features="tfidf", max_features: int = None) -> pd.DataFrame:
    """Encode a Series of text string to TF-IDF.

    :param data: input data
    :type data: pd.Series
    :param features: the features to be outputed. Can take values "tfidf", "count", "both".
    :type features: str
    :param max_features: maximum features limit for CountVectorizer
    :type max_features: int
    :rtype: np.ndarray
    """
    vectorizer = CountVectorizer(max_features=max_features)
    tfidf = TfidfTransformer()
    X = vectorizer.fit_transform(data["processed_text"])
    df_count = pd.DataFrame(
        X.toarray(),
        columns=[f"count_{s}" for s in vectorizer.get_feature_names_out()]
    )
    if features.lower() == "count":
        return df_count
    XX = tfidf.fit_transform(X)
    df_tfidf = pd.DataFrame(
        XX.toarray(),
        columns=[f"tfidf_{s}" for s in vectorizer.get_feature_names_out()]
    )
    if features.lower() == "tfidf":
        return df_tfidf
    elif features.lower() == "both":
        return pd.concat([df_count, df_tfidf], axis=1)
    else:
        raise RuntimeError(
            "Invalid features. features can only take values 'tfidf' or 'count'.")


def word_embedding(
    df: pd.DataFrame,
    vector_size:int=1000,
    w2v_epochs:int=30,
    aggregate:str="mean",
    colname:str="text"
):
    """creates word embedding feature space for dataset.
    
    :param df: input dataframe
    :type df: pd.DataFrame
    :param vector_size: size of embedding vectors
    :type vector_size: int
    :param w2v_epochs: number of epochs to train the Word2Vec transformer on document.
    :type w2v_epochs: int
    :param aggregate: the aggregate function to transform the embedding matrix for each
        sentence to a vector. Can take value of "mean" or "l3"
            - mean: takes average across each word in a sentence
            - l3: take average of the component-wise 3rd power of each vector. This is
                done to prioritize larger values and to preserve (+) or (-) sign.
    :type aggregate: str
    :param colname: name of text column in df
    :type colname: str
    """
    word_vector = []
    text_col = df[colname]
    tokenized_text = [simple_preprocess(line, deacc=True) for line in text_col]
    porter_stemmer = PorterStemmer()
    stemmed_tokens = [[porter_stemmer.stem(
        word) for word in tokens] for tokens in tokenized_text]

    w2v_model = Word2Vec(sentences=stemmed_tokens,
                         vector_size=vector_size, window=5, min_count=1, workers=4, sg=1)

    # below is added to Mike's version

    w2v_model.build_vocab(stemmed_tokens)
    w2v_model.train(
        stemmed_tokens,
        total_examples=len(stemmed_tokens),
        epochs=w2v_epochs
    )

    # above is added to Mike's version

    for index, row in enumerate(stemmed_tokens):
        model_vector = np.zeros((vector_size, len(row)))
        for tok_id, token in enumerate(row):
            if token in w2v_model.wv:
                model_vector[:, tok_id] = w2v_model.wv[token]

        if len(stemmed_tokens[index]) == 0:
            word_vector.append([0]*vector_size)
        else:
            mu = np.mean(model_vector, axis=1)
            m3 = np.mean(model_vector**3, axis=1)
            if aggregate == "mean":
                word_vector.append(mu)
            elif aggregate == "l3":
                word_vector.append(m3)

    # for memory conservation
    minmax = MinMaxScaler()
    df = pd.DataFrame(
        minmax.fit_transform(word_vector), 
        index=None
    )

    df = df.apply(
        pd.to_numeric, downcast="float"
    )
    return df, w2v_model
