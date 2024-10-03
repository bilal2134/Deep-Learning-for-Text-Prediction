# training_script.py

import pandas as pd
import numpy as np
import re
import os
import nltk
import pickle

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Download NLTK stopwords
nltk.download('stopwords')

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Assuming the text is in a column named 'PlayerLine'
    text = ' '.join(data['PlayerLine'].dropna().astype(str))

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Rejoin words to make the cleaned text
    cleaned_text = ' '.join(words)
    return cleaned_text

# Tokenize and prepare sequences
def prepare_sequences(text, max_sequence_len, n_words_to_predict):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Save the tokenizer for later use
    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create input sequences and corresponding outputs
    input_sequences = []
    target_sequences = []
    token_list = tokenizer.texts_to_sequences([text])[0]

    for i in range(max_sequence_len, len(token_list) - n_words_to_predict + 1):
        input_seq = token_list[i - max_sequence_len:i]
        target_seq = token_list[i:i + n_words_to_predict]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    input_sequences = np.array(input_sequences)
    target_sequences = np.array(target_sequences)
    return input_sequences, target_sequences, total_words

# Build the LSTM model
def build_model(total_words, max_sequence_len, n_words_to_predict):
    model = Sequential()
    model.add(Embedding(total_words, 200, input_length=max_sequence_len))
    model.add(LSTM(256))
    model.add(RepeatVector(n_words_to_predict))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(total_words, activation='softmax')))
    model.build(input_shape=(None, max_sequence_len))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# Train the model
def train_model(model, input_sequences, target_sequences, epochs=2, batch_size=64):
    # Reshape target_sequences to (num_samples, n_words_to_predict, 1)
    target_sequences = target_sequences.reshape((target_sequences.shape[0], target_sequences.shape[1], 1))
    target_sequences = target_sequences.astype(np.int32)

    # Print shapes for debugging
    print(f'Input sequences shape: {input_sequences.shape}')
    print(f'Input sequences dtype: {input_sequences.dtype}')
    print(f'Target sequences shape: {target_sequences.shape}')
    print(f'Target sequences dtype: {target_sequences.dtype}')

    # Add early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', save_best_only=True)

    # Train the model
    model.fit(input_sequences, target_sequences, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint])

# Main execution
if __name__ == "__main__":
    file_path = "Shakespeare_data.csv"  # Update with the correct path to your dataset
    max_sequence_len = 30
    n_words_to_predict = 3  # Number of words to predict

    print("Loading and preprocessing data...")
    text = load_and_preprocess_data(file_path)

    print("Preparing sequences...")
    input_sequences, target_sequences, total_words = prepare_sequences(text, max_sequence_len, n_words_to_predict)

    print("Building the model...")
    model = build_model(total_words, max_sequence_len, n_words_to_predict)

    print("Training the model...")
    try:
        train_model(model, input_sequences, target_sequences)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)

    print("Saving the model...")
    model.save('word_prediction_model.keras')

    print("Training complete. Model saved as 'word_prediction_model.h5' and tokenizer saved as 'tokenizer.pkl'")
