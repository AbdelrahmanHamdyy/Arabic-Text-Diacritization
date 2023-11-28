from preprocessing import *

PATH = "../dataset/train.txt"


def extractFeatures():
    sentences = readFile(PATH)
    # 1. Bag of words
    # 2. TF-IDF
    # 3. Word embeddings
    # 4. Trainable embeddings
    # 5. Contextual embeddings
    pass


def readFile(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            preprocessedSentence = preprocess(line.strip())
            print(preprocessedSentence)
            print("------------------------------------------")
            sentences.append(preprocessedSentence)

    return sentences


if __name__ == '__main__':
    extractFeatures()
