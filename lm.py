import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split

'''
Train a LSTM to be a language model on a training split of this dataset. 
Let the model predict next words on sequences of the test set and compare with the original sentences.
'''
#Loading text-file
text_file = "sentences.txt"
with open(text_file, "r", encoding="utf-8") as file:
    sentences = file.readlines()

# Entferne Leerzeichen und leere Zeilen
sentences = [line.strip() for line in sentences if line.strip()] #

#Tokenizer:
tokenizer = Tokenizer(num_words=10000) #reducing possibilities
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
total_words = len(word_index) + 1

#Sequences:
input_sequences = []
for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre') #Padding

#defining x and y
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = pd.get_dummies(y, columns=[0], dtype=int) 

#Training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LSTM-Model
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=L2(0.01))),
    Dropout(0.5),
    LSTM(64, kernel_regularizer=L2(0.01)),
    Dense(64, activation='relu'),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) #Compiling
model.summary()

#Training
epochs = 10 
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

#Prediction function
def predict_next_words(seed_text, num_words=1):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Test: Vorhersage auf einem Beispiel
test_sentence = "Gib hier einen Testsatz ein"
print("Original Sentence:", test_sentence)
predicted_sentence = predict_next_words(test_sentence, num_words=5)
print("Predicted Sentence:", predicted_sentence)

# Ich habs versucht aber nicht hingekriegt. Habe zur Unterstützung ChatGPT genutzt. 
