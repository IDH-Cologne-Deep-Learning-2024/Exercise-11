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

# data import
df = pd.read_fwf("sentences.txt", header=None, names=["sentence"])
df.dropna(inplace=True)

# im not sure why the tokenizer splits the tokens weirdly sometimes
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["sentence"])
total_words = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(df["sentence"])
max_length = max([len(sequence) for sequence in sequences])

# data preprocessing
data = []
labels = []

for seq in sequences:
    for i in range(1, len(seq)):
        data.append(seq[:i])
        labels.append(seq[i])

data = pad_sequences(data, padding="post")
labels = to_categorical(labels, num_classes=total_words)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=40
)

# LSTM Model
LSTM = Sequential(
    [
        Embedding(total_words, 400, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(total_words, activation="softmax"),
    ]
)

LSTM.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.01),
    metrics=["accuracy"],
)
LSTM.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Predict a word from a made up sequence
input_seq = "The yellow car is"
input_tokens = tokenizer.texts_to_sequences([input_seq])[0]
input = pad_sequences([input_tokens], maxlen=max_length - 1, padding="pre")
predicted = LSTM.predict(input)

predicted_word = ""
for word, index in tokenizer.word_index.items():
    if index == np.argmax(predicted[0]):
        predicted_word = word
        break
print("Predicting a word based of a made up sequence.")
print(f"The new input sequence is: {input_seq}")
print(f"The predicted word is: {predicted_word}")


# Predict a word from a test set sequence
# Reversing a sequence from X_test didnt work for me

file = open("sentences.txt", "r")
input_seq = file.readline(50)
file.close()

input_tokens = tokenizer.texts_to_sequences([input_seq])[0]
input = pad_sequences([input_tokens], maxlen=max_length - 1, padding="pre")

predicted = LSTM.predict(input)
predicted_word = ""
for word, index in tokenizer.word_index.items():
    if index == np.argmax(predicted[0]):
        predicted_word = word
        break

print("Predicting a word based of a test set sequence.")
print(f"The test set input sequence is: {input_seq}")
print(f"The predicted word is: {predicted_word}"
