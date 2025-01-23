import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def create_overlap(sequences, seq_length=50, overlap=5):
    X = []
    y = []
    for sequence in sequences:
        for i in range(0, len(sequence) - seq_length, overlap):
            group = sequence[i:i+seq_length]
            next_word = sequence[i + seq_length]
            X.append(group)
            y.append((next_word))
    return (np.array(X), to_categorical(y, vocab_size))


def make_prediction(X, y, model):
    for seq, y_value in zip(X, y):
        seq_as_words = " ".join([tokenizer.index_word[s] for s in seq])
        pred = model.predict(seq[np.newaxis, :], verbose=0)
        vocab_index = np.argmax(pred)
        gold_word = tokenizer.index_word[np.argmax(y_value)]
        pred_word = tokenizer.index_word[vocab_index]
        print(f"Gold Words: {seq_as_words} {gold_word}")
        print(f"Pred Words: {seq_as_words} {pred_word}")
        print()


with open("sentences.txt", "r") as fo:
    data = fo.read().split("\n")
data = data[0:1000]
data_train, data_test = train_test_split(data, test_size=0.2)
X_train = [" ".join(data_train)]
X_test = [" ".join(data_test)]
print(X_test)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([" ".join(data)])
vocab_size = len(tokenizer.word_index)+1
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
seq_length = 50
X_train, y_train = create_overlap(X_train, seq_length=seq_length, overlap=5)
X_test, y_test = create_overlap(X_test, seq_length=seq_length, overlap=5)
print(X_train)
print(y_train)

model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=seq_length))
#model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.2, verbose=1)
make_prediction(X_test, y_test, model)
