from preprocessing import *
from utils import *
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = "../dataset/train.txt"
corpus = readFile(PATH)


# 1. Bag of words
# 2. TF-IDF
# 3. Word embeddings
# 4. Trainable embeddings
# 5. Contextual embeddings


def extractWordEmbeddings():
    data = []
    for sentence in corpus:
        sentence = word_level_preprocess(sentence)
        data.append(sentence)
    model = Word2Vec(data, vector_size=100, min_count=1)
    print(model.wv['الْبَلَدِ'])
    print(model.wv.most_similar('الْبَلَدِ'))
    model.save('models/w2vmodel')


def TF_IDF():
    data = []
    for sentence in corpus:
        sentence = data_cleaning(sentence)
        data.append(sentence)
    tr_idf_model = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(data)
    tf_idf_array = tf_idf_vector.toarray()
    words_set = tr_idf_model.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns=words_set)
    print(df_tf_idf)


if __name__ == '__main__':
    extractWordEmbeddings()
    TF_IDF()
