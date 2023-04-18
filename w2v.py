from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

def word_embedding(
        df: pd.DataFrame, 
        vector_size=1000, 
        w2v_epochs = 30,
        aggregate = "mean",
        colname = "text"
    ):
    word_vector = []
    text_col = df[colname]
    tokenized_text = [simple_preprocess(line, deacc=True) for line in text_col]
    porter_stemmer = PorterStemmer()
    stemmed_tokens = [[porter_stemmer.stem(word) for word in tokens] for tokens in tokenized_text]
    
    w2v_model = Word2Vec(sentences=stemmed_tokens, vector_size=vector_size, window=5, min_count=1, workers=4, sg=1)
    
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
            mu = np.mean(model_vector, axis = 1)
            m3 = np.mean(model_vector**3, axis = 1)
            if aggregate == "mean":
                word_vector.append(mu)
            elif aggregate == "l3":
                word_vector.append(m3)

    return pd.DataFrame(word_vector, index=None)