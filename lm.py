import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataframe = pd.read_fwf("sentences.txt", header = None, names = ["sentence"])
dataframe.dropna(inplace=True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataframe["sentence"])
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(dataframe["sentence"])

X, y = [], []

for seq in sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])

X = pad_sequences(X, padding="post")
y = to_categorical(y, num_classes = vocab_size)

#train_size = int(0.8 * len(X))
#X_train, X_test = X[:train_size], X[train_size:]
#y_train, y_test = y[:train_size], y[train_size:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


hidden_size = 128
model = Sequential()
model.add(Embedding(input_dim = vocab_size + 1, output_dim=128))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.45))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.45))
model.add(LSTM(hidden_size))
model.add(Dropout(0.45))
model.add(Dense(vocab_size, activation="sigmoid"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6)

model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),  callbacks=[early_stopping, reduce_lr])

def predict_next_word(model, tokenizer, text, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen = max_sequence_length, padding = "pre")
    
    predicted_prob = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(predicted_prob)
    
    predicted_word = tokenizer.index_word[predicted_index]
    
    return predicted_word

f = open("sentences.txt", "r")

random_number = np.random.randint(low=0, high=vocab_size)
#print(vocab_size)
#print(random_number)

def pick_sentence(number):
    return f.readline(number)

test_sequence = pick_sentence(random_number)

predicted_word = predict_next_word(model, tokenizer, test_sequence, X.shape[1])
print(f"Predicted next word: {predicted_word}")