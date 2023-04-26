"""
IEOR4525 Machine Learning For FE and OR - Twitter Sentiment Analysis, Spring 2023

Functional module. Containing all utility functions.
"""

import feature_engineering as fe
import my_globals
import numpy as np
import pandas as pd

from preprocessing import cleaning, preprocess_pipeline

# Always save main dataset outside of the code repo.
# In this case, my data_dir is "../project_data" parallel to "."


def get_sub_dataset(size: int = 5000, random_seed: int = 0) -> pd.DataFrame:
    """Generate a subset of the main dataset usinga random seed.
    Saves sub-dataset to a diectory outside of and parallel to main directory. 

    :param size: size of subset
    :type size: int, optional. Default to be 5000.
    :param random_seed: random seed
    :type random_seed: int, optional. Default to be 0.
    :rtype: pd.DataFrame
    """
    data = get_entire_dataset()
    np.random.seed(random_seed)
    idx = np.random.choice(data.index, size)
    data = data.loc[idx]
    data.to_csv(
        "/".join([my_globals.DATA_DIR, f"twitter_seed{random_seed}.csv"]),
        index=False,
        encoding="latin1"
    )
    return data.reset_index()


def get_sub_featured_datasets(
    size: int = 5000,
    random_seed: int = 0,
    pipeline: str = "conservative",
    max_features: int = None
) -> pd.DataFrame:
    """Generate a subset of the main dataset and conduct feature engineering.

    :param size: size of subset
    :type size: int, optional. Default to be 5000.
    :param random_seed: random seed
    :type random_seed: int, optional. Default to be 0.
    :param pipeline: the pipelining style applied in text processing.
        Either "concervative" or "aggresive".
    :type pipeline: str
    :rtype: pd.DataFrame
    """

    data = get_sub_dataset(size, random_seed)
    data = cleaning(data)
    data["processed_text"] = data["text"].apply(
        lambda s:
        preprocess_pipeline(
            s,
            pipeline=pipeline
        )
    )
    data["exclaim_freq"] = data["text"].apply(fe.exclaim_freq)
    data["mention_count"] = data["text"].apply(fe.mention_count)
    data["cap_freq"] = data["text"].apply(fe.cap_freq)
    count_tfidf = fe.get_token_features(
        data,
        features="tfidf",
        max_features=max_features
    )
    feature_columns = ["exclaim_freq", "mention_count", "cap_freq"]
    feature_columns += [col for col in data.columns if "weekday" in col]
    data = pd.concat([data[feature_columns], count_tfidf], axis=1)
    return data


def get_entire_dataset() -> pd.DataFrame:
    """Fetch the entire dataset as a dataframe.

    :rtype: pd.DataFrame
    """
    DATA_PATH = "/".join([my_globals.DATA_DIR, my_globals.MAIN_DATA_NAME])
    data = pd.read_csv(
        DATA_PATH,
        # encoding = 'ISO-8859-1',
        encoding="latin1",
        header=None,
        names=my_globals.HEADER
    )
    return data


def get_feature_space(
        N: int,
        feature_space: int, 
        max_features: int, 
        w2v_aggregate: str = "l3",
        random_seed:int = 0
    ):
    """Executer to get full feature spaces 1 or 2.

    :param N: size of data subset
    :type N: int
    :param feature_space: feature space 1 or 2.
    :type feature_space: int
    :param max_features: the maximum number of features. 
        For F1, it's the max_feature parameter for CountVectorizer;
        For F2, it's the length of embedding vectors.
    :type max_features: int
    :param w2v_aggregate: the aggregate function to be applied onto output tensor
        of the embedding step to transform it to a 2d matrix.
        Takes value "l3" or "mean".
    :type w2v_aggregate: str, optional, default "l3"
    :param random_seed: random seed.
    :type random_seed: int
    """

    if feature_space not in [1, 2]:
        raise RuntimeError("feature_space = 1 or 2.")
    data = get_sub_dataset(size = N, random_seed=random_seed)
    data = cleaning(data)

    if feature_space == 1:
        data["processed_text"] = data["text"].apply(
            lambda s:
            preprocess_pipeline(
                s,
                pipeline="conservative"
            )
        )

        feature_df = fe.get_token_features(
            data,
            features="tfidf",
            max_features=max_features
        )

    elif feature_space == 2:
        data["processed_text"] = data["text"].apply(
            lambda s:
            preprocess_pipeline(
                s,
                pipeline="w2v"
            )
        )
        feature_df, _ = fe.word_embedding(
            data,
            vector_size=max_features,
            w2v_epochs=30,
            aggregate=w2v_aggregate,
            colname="processed_text"
        )

    data["exclaim_freq"] = data["text"].apply(fe.exclaim_freq)
    data["mention_count"] = data["text"].apply(fe.mention_count)
    data["cap_freq"] = data["text"].apply(fe.cap_freq)
    feature_columns = ["exclaim_freq", "mention_count", "cap_freq", "target"]
    feature_columns += [col for col in data.columns if "weekday" in col]
    data = pd.concat([data[feature_columns], feature_df], axis=1)
    return data

