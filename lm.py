import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#get data
dataframe = pd.read_fwf('sentences.txt', header=None , names=['sentences'])
dataframe.dropna(inplace=True)
#tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataframe['sentences'])
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(dataframe['sentences'])
#padding
X = pad_sequences(sequences, maxlen=vocab_size, padding='post')
#print(X)
y = np.random.randint(2, size=(X.shape[0], 1))
#print(y)

#preparing for model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=43)

output_dim = 256

#verhindern von overfitting val_loss = misst den Verlust auf den Validierungsdaten, wert wie gut ungesehene daten functionieren 
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1) 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1)

#model
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=output_dim))
model.add(LSTM(output_dim, return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(output_dim, return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(output_dim))
model.add(Dropout(0.15))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=45, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

#prediction

def predict_next_word(model, tokenizer, text, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen = max_sequence_length, padding = "pre")

    print(tokenizer.index_word)
    print(tokenizer.word_index)
    
    predicted_prob = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(predicted_prob)
    
     # Vorhersage
    predicted_probs = model.predict(sequence)
    print(predicted_probs)
    predicted_index = np.argmax(predicted_probs)
    print(predicted_index)
    print(sequence)

     # Index überprüfen
    if predicted_index == 0:
        return "<PAD>"  # Für Padding-Tokens
    elif predicted_index in tokenizer.index_word:
        return tokenizer.index_word[predicted_index]
    else:
        return "<UNK>"  # Für unbekannte Wörter
    


random_number = np.random.randint(low=0, high=vocab_size)

f = open("sentences.txt", "r")
test_sequence = f.readline(random_number)
f.close()

predicted_word = predict_next_word(model, tokenizer, test_sequence, X.shape[1])
print(f"Predicted next word: {predicted_word}")


  

