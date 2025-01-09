import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split

df = pd.read_fwf("sentences.txt", header=None, names=["sentence"])
df.dropna(inplace=True)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["sentence"])
sequences = tokenizer.texts_to_sequences(df["sentence"])

max_sequence_length = max([len(sequence) for sequence in sequences])
data = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")
labels = np.random.randint(2, size=(data.shape[0], 1))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(), loss="binary_crossentropy")

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))