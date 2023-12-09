import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

PATH = "../dataset/train.txt"
corpus = readFile(PATH)
def RNN():
    # Create sequences using Word2Vec embeddings
    X_train_sequences = [[word2vec_model.wv[word] for word in sentence] for sentence in X_train]
    X_test_sequences = [[word2vec_model.wv[word] for word in sentence] for sentence in X_test]

    # Pad sequences for uniform length
    X_train_pad = pad_sequences(X_train_sequences)
    X_test_pad = pad_sequences(X_test_sequences, maxlen=X_train_pad.shape[1])

    # Build RNN model
    model = Sequential()
    model.add(Embedding(input_dim=len(word2vec_model.wv.index_to_key) + 1, output_dim=100, input_length=X_train_pad.shape[1]))
    model.add(SimpleRNN(units=128))
    model.add(Dense(units=len(set(encoded_labels)), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')





if __name__ == '__main__':
    RNN()