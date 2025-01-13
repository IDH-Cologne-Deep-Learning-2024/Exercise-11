import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

data = pd.read_csv('path/to/your/file.csv', header=None, names=['sentence'])
sentences = data['sentence'].astype(str).tolist()

max_words = 10000  
max_sequence_length = 20 

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index

input_sequences = []
next_words = []
for seq in sequences:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i])
        next_words.append(seq[i])

X = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
y = np.array(next_words)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_dim = 128
hidden_units = 128

model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(hidden_units, return_sequences=False),
    Dropout(0.2),
    Dense(max_words, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

def predict_next_word(seed_text, num_words):
    for _ in range(num_words):
        tokenized_sequence = tokenizer.texts_to_sequences([seed_text])
        tokenized_sequence = pad_sequences(tokenized_sequence, maxlen=max_sequence_length, padding='pre')
        predicted_word_index = np.argmax(model.predict(tokenized_sequence), axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                seed_text += ' ' + word
                break
    return seed_text

for i in range(5): 
    seed_text = ' '.join([tokenizer.index_word[idx] for idx in X_test[i] if idx != 0])
    original_sentence = ' '.join([tokenizer.index_word[idx] for idx in X_test[i][1:] if idx != 0])
    predicted_sentence = predict_next_word(seed_text, num_words=1)
    print(f"Seed text: {seed_text}")
    print(f"Original sentence: {original_sentence}")
    print(f"Predicted sentence: {predicted_sentence}")
    print("-" * 50)
