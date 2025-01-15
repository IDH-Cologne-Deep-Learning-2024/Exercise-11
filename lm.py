import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
with open("sentences.txt", "r", encoding="utf-8") as file:
    df = file.read().split('<eos>')

df = [sentence.strip() for sentence in df if sentence.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df)
sequences = tokenizer.texts_to_sequences(df)

X = [seq[:-1] for seq in sequences if len(seq) > 1]
y = [seq[1:] for seq in sequences if len(seq) > 1]

MAX_LENGTH = max(max(len(seq) for seq in X), max(len(seq) for seq in y))
X = pad_sequences(X, maxlen=MAX_LENGTH, padding='post')
y = pad_sequences(y, maxlen=MAX_LENGTH, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=136, input_length=MAX_LENGTH))
model.add(LSTM(132, return_sequences=True))
model.add(Dropout(0.4))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)
#w/o prediction