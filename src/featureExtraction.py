from preprocessing import *
from utils import *

PATH = "../dataset/train.txt"


def extractFeatures():
    sentences = readFile(PATH)
    print(sentences)
    # 1. Bag of words
    # 2. TF-IDF
    # 3. Word embeddings
    # 4. Trainable embeddings
    # 5. Contextual embeddings
    pass


if __name__ == '__main__':
    extractFeatures()
