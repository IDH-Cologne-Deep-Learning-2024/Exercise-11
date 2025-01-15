import numpy as np
from keras.src.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from keras.src.utils import to_categorical, pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import L1, L2, L1L2


# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        predictions = model.predict(encoded, verbose=0)
        yhat = np.argmax(predictions, axis=1)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text


with open("sentences.txt", encoding="utf8") as file:
    data = file.read()
all_tokens = data.split()  # Split on each whitespace like \t \n " " etc.

# Remove chars like: <> and . form filters, to include eos token and punctuation as single tokens
tokenizer = Tokenizer(filters="\t\n")
tokenizer.fit_on_texts(all_tokens)
encoded = tokenizer.texts_to_sequences(all_tokens)
vocab_size = len(tokenizer.word_index) + 1  # Padding-Token = 0, therefore + 1
eos_token = tokenizer.word_index["<eos>"]

# Create the list for X and y with sliding window approach, set to window_size = 4 (3 tokens for X followed by the y
# token)
X = []
y = []
i = 0
len_encoded = len(encoded)
window_size = 4
while i + window_size <= len_encoded:
    X_line = []
    for j in range(window_size):
        if j != window_size - 1:
            X_line.append(encoded[i + j])
        else:
            y.append(encoded[i + j])
    # If <eos> is encountered next window will start after it (to prevent <eos> being part of X)
    if encoded[i + window_size - 1] == eos_token:
        i = i + window_size
    else:
        i += 1
    # Shape the X_line data to 1D array and append to X
    X.append(np.array(X_line).reshape(3, ))

X = np.array(X)
y = np.array(y)

MAX_LENGTH = max(len(x) for x in X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# Compile network
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Fit network
model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=20)
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

# Check sequence generation with own input
print(generate_seq(model, tokenizer, MAX_LENGTH, 'the green', 5))
