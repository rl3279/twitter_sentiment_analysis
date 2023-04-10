import my_globals
import pandas as pd
import numpy as np
from typing import List
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import contractions
from functools import reduce
from dateutil.parser import parse


def setup_nltk():
    """Downloads necessary packages within nltk"""
    packages = ["punkt", "wordnet", "stopwords"]
    for p in packages:
        try:
            nltk.data.find(p)
        except LookupError:
            nltk.download(p)


def tokenize(s: str, how="word_tokenize") -> List[str]:
    if how == "word_tokenize":
        return word_tokenize(s)
    elif how == "split":
        return s.split()


def del_username(s: str) -> str:
    """Delete @Username from a tweet str.

    :param s: input string
    :type s: str
    :rtype: str
    """

    return " ".join([t for t in tokenize(s, how="split") if not t.startswith("@")])


def del_punc(s: str) -> str:
    """Delete punctuations from str.

    :param s: input string
    :type s: str
    :rtype: str
    """
    punc = my_globals.PUNCS
    return "".join([w for w in s if w not in punc])


def del_link(s: str) -> str:
    """Delete links from str.

    :param s: input string
    :type s: str
    :rtype: str
    """
    r = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)"
    return " ".join([re.sub(r, "", t) for t in tokenize(s, how="split")])


def decontract(s: str) -> str:
    """Remove contractions in text.
    e.g. I'm -> I am; she'd -> she would

    :param s: input string
    :type s: str
    :rtype: str
    """
    tokens = []
    for t in tokenize(s, how="split"):
        tokens.append(contractions.fix(t))
    return " ".join(tokens)


def del_stopwords(s: str) -> str:
    """Delete stopwords and punctuation from a string.
    Note that the type-hinting indicates that this function ought
    to be run first in the pre-processing pipeline.

    :param s: input string
    :type s: str
    """
    stop_words = set(stopwords.words('english'))

    return " ".join([t for t in tokenize(s) if t not in stop_words])

def remove_digits(s: str) -> bool:
    """Detect digits from str.

    :param s: input string
    :type s: str
    :rtype: bool
    """
    cleaned_string = re.sub(r'\w*\d\w*|[^\w\s]', '', s)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    cleaned_string = cleaned_string.strip()

    return bool(cleaned_string)

def del_digits(s: str) -> str:
    """Delete digits from str.

    :param s: input string
    :type s: str
    :rtype: str
    """
    return " ".join([w for w in tokenize(s) if remove_digits(w)])

def lemmatize(s: str) -> str:
    """Lemmatize str.

    :param s: input string
    :type s: str
    :rtype: str
    """
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(t) for t in tokenize(s)])


def preprocess_pipeline(s: str, return_lower = True) -> str:
    """Run string through all pre-processing functions.

    :param s: input string
    :type s: str
    :rtype: str
    """
    s = reduce(
        lambda value, function: function(value),
        (
            del_link,
            del_username,
            decontract,
            lemmatize,
            del_stopwords,
            del_punc,
            del_digits
        ),
        s,
    )

    return s.lower() if return_lower else s

def str_datetime(s: str):
    """Parse and format a datetime str to weekday and datetime.MAXYEAR
    
    :param s: input string containing datetime information
    :type s: str
    :rtype: tuple[str, str]
    """
    ss = parse(s).strftime('%a %Y-%m-%d %H:%M:%S')
    return ss[:3], ss[4:]


def cleaning(df: pd.DataFrame):
    """Cleaning script of the data (or subset).
    
    :param df: input dataframe.
    :type df: pd.DataFrame
    :rtype: pd.DataFrame
    """
    # Parse weekday and datetime
    weekday_datetime = pd.DataFrame(
        list(df.loc[:, "date"].apply(str_datetime)),
        columns=["weekday", "datetime"]
    )
    # One-hot encode weekday
    weekdaydummies = pd.get_dummies(
        weekday_datetime['weekday'], 
        prefix='weekday', 
        dtype=float
    )
    weekdaydummies = pd.DataFrame(
        weekdaydummies, 
        columns=['weekday_'+w for w in [
            "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
        ]]
    )
    # Concatenate weekday dummies to other features
    weekdaydummies_datetime = pd.concat(
        [weekdaydummies, weekday_datetime['datetime']], 
        axis=1
    )
    df = pd.concat([df, weekdaydummies_datetime], axis=1)
    # Drop the column with single unique value.
    df.drop("flag", axis = 1, inplace=True)
    return df