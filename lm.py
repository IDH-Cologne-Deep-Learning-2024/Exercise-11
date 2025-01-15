import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 1. Load the sentences from the text file
with open("sentences.txt", "r", encoding="utf-8") as file:
    sentences = file.readlines()

# Remove newline character from each sentence
sentences = [sentence.strip() for sentence in sentences]

# Tokenize the sentences into sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Convert the sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(sentences)

# Find the maximum sequence length
max_sequence_length = max(len(seq) for seq in sequences)

# Pad the sequences to have equal length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="pre")

# Prepare X and y (input and target)
X = padded_sequences[:, :-1]  # Input: all words except the last one
y = padded_sequences[:, -1]   # Output: the last word in the sentence

# One-hot encode the target variable (y)
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the LSTM model
model = Sequential()

# Embedding layer to convert words into dense vectors
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,  # Vocabulary size
                    output_dim=100,  # Size of word embeddings (vector dimension)
                    input_length=max_sequence_length - 1))  # Length of the input sequence

# LSTM layer for sequence modeling
model.add(LSTM(128, return_sequences=False))

# Dense layer for output prediction (softmax for multiclass classification)
model.add(Dense(len(tokenizer.word_index) + 1, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model.summary()

# 3. Model training
# Train the LSTM model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 4. Generate predictions
# Function to predict the next word given a sequence
def predict_next_word(model, tokenizer, text, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length - 1, padding="pre")

    predicted = model.predict(padded_sequence)
    predicted_word_index = np.argmax(predicted)

    # Get the word corresponding to the predicted index
    predicted_word = tokenizer.index_word.get(predicted_word_index, "")
    return predicted_word

# Test with a sample sequence
test_sentence = "This is a great"
predicted_word = predict_next_word(model, tokenizer, test_sentence, max_sequence_length)
print(f"Predicted next word for '{test_sentence}': {predicted_word}")
