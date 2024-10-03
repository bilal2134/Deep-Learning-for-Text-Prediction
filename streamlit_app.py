# streamlit_app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model and tokenizer
model = load_model('word_prediction_model.keras')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set parameters
max_sequence_len = 30
n_words_to_predict = 3

def predict_next_words(model, tokenizer, text, max_sequence_len, n_words_to_predict):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(sequence, verbose=0)
    predicted_word_indices = np.argmax(predicted[0], axis=1)
    output_words = [tokenizer.index_word.get(index, '') for index in predicted_word_indices]
    return ' '.join(output_words)

st.title("Next Word Prediction")

text = st.text_input("Enter your text:")

if text:
    next_words = predict_next_words(model, tokenizer, text, max_sequence_len, n_words_to_predict)
    st.write(f"Next words prediction: {next_words}")
