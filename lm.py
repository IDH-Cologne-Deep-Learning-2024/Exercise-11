import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import tensorflow as tf

# Set working directory
def set_working_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

# Function to load and preprocess dataset
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    # Remove trailing spaces and <eos> tokens
    sentences = [sentence.strip().replace(' <eos>', '') for sentence in sentences]
    return sentences

# Function to tokenize and generate input-output pairs
def tokenize_and_generate_pairs(sentences, tokenizer):
    sequences = tokenizer.texts_to_sequences(sentences)
    
    data = []
    for seq in sequences:
        for i in range(1, len(seq)):
            input_seq = seq[:i]
            output_word = seq[i]
            data.append((input_seq, output_word))

    return data

# Function to pad sequences and prepare data
def prepare_data(data, tokenizer):
    X, y = zip(*data)

    max_seq_len = max(len(seq) for seq in X)
    X = pad_sequences(X, maxlen=max_seq_len, padding='pre')

    y = np.array(y) - 1  # Convert to zero-based index
    vocab_size = len(tokenizer.word_index)

    return X, y, max_seq_len, vocab_size

# Function to build the LSTM model
def build_lstm_model(vocab_size, max_seq_len):
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=256, input_length=max_seq_len),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.4),
        LSTM(256, return_sequences=False),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to calculate perplexity
def calculate_perplexity(loss):
    return np.exp(loss)

# Improved Beam Search Prediction
def predict_with_beam_search(model, tokenizer, seed_text, max_len, beam_width=3):
    sequences = [(seed_text, 0.0)]  # List of tuples (sequence, score)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            tokenized_seq = tokenizer.texts_to_sequences([seq])[0]
            padded_seq = pad_sequences([tokenized_seq], maxlen=max_len, padding='pre')

            preds = model.predict(padded_seq, verbose=0)[0]
            top_indices = np.argsort(preds)[-beam_width:]

            for idx in top_indices:
                word = tokenizer.index_word.get(idx + 1, "<UNK>")
                candidate = seq + ' ' + word
                candidate_score = score - np.log(preds[idx])
                all_candidates.append((candidate, candidate_score))

        # Prune candidates
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    return sequences[0][0]

# Main script execution
def main():
    set_working_directory()

    file_path = "sentences.txt"
    sentences = load_and_preprocess_data(file_path)

    # Tokenizer setup
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(sentences)

    data = tokenize_and_generate_pairs(sentences, tokenizer)
    X, y, max_seq_len, vocab_size = prepare_data(data, tokenizer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model(vocab_size, max_seq_len)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=64, 
                        validation_split=0.2, 
                        callbacks=[early_stopping, lr_scheduler])

    loss, accuracy = model.evaluate(X_test, y_test)
    perplexity = calculate_perplexity(loss)

    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}, Test Perplexity: {perplexity}')

    for seed_sentence in sentences[:5]:
        seed_text = ' '.join(seed_sentence.split()[:3])
        generated_text = predict_with_beam_search(model, tokenizer, seed_text, max_len=10)
        print(f"Seed: {seed_text}")
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
