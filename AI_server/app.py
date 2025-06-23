from flask import Flask, request, jsonify
import threading
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

MODEL_PATH = "output/sentiment_model_keras.h5"
VOCAB_PATH = "output/tfidf_vocab.json"
LABELS_PATH = "output/label_classes.json"

# Load model and artifacts
model = None
tfidf_vectorizer = None
labels = None

def load_artifacts():
    global model, tfidf_vectorizer, labels
    model = load_model(MODEL_PATH)

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    # inject loaded vocabulary for inference only
    tfidf_vectorizer.vocabulary_ = vocab

    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)

load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({"error": "Missing 'sentence' field"}), 400

    # Vectorize input
    X = tfidf_vectorizer.transform([sentence]).toarray()
    preds = model.predict(X)
    pred_label = labels[np.argmax(preds)]

    return jsonify({"prediction": pred_label})

@app.route('/train', methods=['POST'])
def train():
    from subprocess import Popen

    dataset_path = request.json.get('dataset_path', 'training_dataset.csv')

    # Run training asynchronously
    def run_training():
        Popen(['python3', 'sentiment_analysis_training.py', dataset_path]).wait()
        load_artifacts()  # reload model and vocab after training

    threading.Thread(target=run_training).start()

    return jsonify({"message": "Training started"}), 202

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
