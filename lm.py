import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split

data_path = "sentences.txt"
with open(data_path, 'r') as f:
    sentences = f.readlines()

train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentences)
vocab_size = len(tokenizer.word_index) + 1  

train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

max_seq_length = max(len(seq) for seq in train_sequences)


train_sequences = pad_sequences(train_sequences, maxlen=max_seq_length, padding='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post')

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]
X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100),
    LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2), 
    Dense(vocab_size, activation='softmax')
])
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=32])