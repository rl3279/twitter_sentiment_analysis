DATA_DIR = "/Users/zhengyizhu/Desktop/23spring/4525 ML/data"
MAIN_DATA_NAME = "raw-data.csv"
HEADER = ["target", "ids", "date", "flag", "user", "text"]
STOPWORDS_PATH = DATA_DIR + "/stopwords.txt"
PUNCS = """"#$%&\'()*+,-./:;[\\]^_{|}~`!?"""
PREPROCESSING_PIPELINE_ORDER = [del_username, decontract, del_strange_characters, del_stopwords, lemmatize]