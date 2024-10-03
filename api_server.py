# api_server.py

from flask import Flask, request, jsonify
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

app = Flask(__name__)

def predict_next_words(model, tokenizer, text, max_sequence_len, n_words_to_predict):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(sequence, verbose=0)
    predicted_word_indices = np.argmax(predicted[0], axis=1)
    output_words = [tokenizer.index_word.get(index, '') for index in predicted_word_indices]
    return ' '.join(output_words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        next_words = predict_next_words(model, tokenizer, text, max_sequence_len, n_words_to_predict)
        return jsonify({'next_words': next_words})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
