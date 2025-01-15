import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split

df = pd.read_csv("sentences.txt", header=None, names=['sentence'])
df.dropna(inplace=True)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["sentence"])
sequences = tokenizer.texts_to_sequences(df["sentence"])

max_sequence_length = 20
X = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")
y_next = np.array([seq[1:] + [0] for seq in sequences])  
y_binary = np.random.randint(2, size=(X.shape[0], 1)) 

X_train, X_test, y_train_next, y_test_next = train_test_split(X, y_next, test_size=0.2, random_state=42)
_, _, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)

input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length)(input_layer)


lstm_layer_next = LSTM(128)(embedding_layer)
dropout_next = Dropout(0.2)(lstm_layer_next)
output_next = Dense(10000, activation='softmax', name='next_word_output')(dropout_next)


lstm_layer_binary = LSTM(128)(embedding_layer)
dropout_binary = Dropout(0.2)(lstm_layer_binary)
output_binary = Dense(1, activation='sigmoid', name='binary_output')(dropout_binary)


model = Model(inputs=input_layer, outputs=[output_next, output_binary])


model.compile(
    optimizer='adam',
    loss={'next_word_output': 'sparse_categorical_crossentropy', 'binary_output': 'binary_crossentropy'},
    metrics={'next_word_output': 'accuracy', 'binary_output': 'accuracy'}
)


model.fit(
    X_train,
    {'next_word_output': y_train_next, 'binary_output': y_train_binary},
    validation_data=(X_test, {'next_word_output': y_test_next, 'binary_output': y_test_binary}),
    epochs=10,
    batch_size=64
)

# Melissa Eleanya :D

