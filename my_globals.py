"""
IEOR4525 Machine Learning For FE and OR - Twitter Sentiment Analysis, Spring 2023

Global variable module. Contains all static variables shared across the repo.
"""

DATA_DIR = "../project_data"
DATA_SIZE = 1600000
MAIN_DATA_NAME = "training.1600000.processed.noemoticon.csv"
HEADER = ["target", "ids", "date", "flag", "user", "text"]
STOPWORDS_PATH = DATA_DIR + "/stopwords.txt"
PUNCS = """"#$%&\'()*+,-./:;[\\]^_{|}~`!?"""
PIPELINE = {
    "aggresive": [
        "del_username", "decontract", "del_strange_characters",
        "del_stopwords", "lemmatize", "del_awww"
    ],
    "conservative": [
        "del_link", "del_username", "decontract",
        "lemmatize", "del_stopwords", "del_punc",
        "del_digits", "del_awww"
    ],
    "w2v": [
        "del_link", "del_username", "decontract",
        "lemmatize"
    ]
}
