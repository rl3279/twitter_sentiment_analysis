from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec

def word_embedding(df: pd.DataFrame, vector_size=1000):
    word_vector = []
    
    tokenized_text = [simple_preprocess(line, deacc=True) for line in df['text']]
    porter_stemmer = PorterStemmer()
    stemmed_tokens = [[porter_stemmer.stem(word) for word in tokens] for tokens in tokenized_text]
    
    w2v_model = Word2Vec(sentences=stemmed_tokens, vector_size=vector_size, window=5, min_count=1, workers=4, sg=1)
    
    for index, row in enumerate(stemmed_tokens):
        model_vector = np.zeros(vector_size)
        for token in stemmed_tokens[index]:
            if token in w2v_model.wv:
                model_vector += w2v_model.wv[token]
        
        if len(stemmed_tokens[index]) == 0:
            word_vector.append([0 for i in range(vector_size)])
        else:
            model_vector /= len(stemmed_tokens[index])  # Calculate the average
            word_vector.append(model_vector)
    
    return pd.DataFrame(word_vector, index=None)