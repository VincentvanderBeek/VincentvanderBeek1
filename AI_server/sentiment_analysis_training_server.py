import pandas as pd
import numpy as np
import os
import json
import logging
import traceback
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

OUTPUT_DIR = "output"

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def train_model(dataset_path='training_dataset.csv'):
    try:
        logging.info(f"Using dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)

        if 'sentence' not in df.columns or 'emotion' not in df.columns:
            logging.error("Dataset must include 'sentence' and 'emotion' columns.")
            return False

        df.dropna(subset=['sentence', 'emotion'], inplace=True)
        logging.info(f"Loaded {len(df)} valid rows")

        ensure_output_dir()

        # Vectorize sentences with TF-IDF
        tfidf_sentence = TfidfVectorizer(max_features=3000, stop_words='english')
        X = tfidf_sentence.fit_transform(df['sentence']).toarray()  # dense needed for Keras

        # Encode labels to integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['emotion'])
        num_classes = len(label_encoder.classes_)
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )

        input_dim = X_train.shape[1]

        # Build a simple feedforward NN
        model = Sequential([
            Dense(256, input_dim=input_dim, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        logging.info("Training Keras model...")
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test accuracy: {test_acc:.4f}")

        # Predictions for report
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

        # Save model to .h5
        model.save(os.path.join(OUTPUT_DIR, "sentiment_model_keras.h5"))
        logging.info("Keras model saved as sentiment_model_keras.h5")

        # Save TF-IDF vectorizer vocab for inference
        with open(os.path.join(OUTPUT_DIR, "tfidf_vocab.json"), "w") as f:
            json.dump(tfidf_sentence.vocabulary_, f)
        logging.info("TF-IDF vocabulary saved to tfidf_vocab.json")

        # Save label encoder classes
        with open(os.path.join(OUTPUT_DIR, "label_classes.json"), "w") as f:
            json.dump(label_encoder.classes_.tolist(), f)
        logging.info("Label classes saved to label_classes.json")

        # Save config with performance and label info
        config = {
            "test_accuracy": test_acc,
            "labels": label_encoder.classes_.tolist(),
            "model_type": "Keras Sequential NN",
            "features": {
                "sentence_tfidf": "max_features=3000, stop_words='english'"
            },
            "notes": "Input vector is dense TF-IDF from sentence only."
        }
        with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        logging.info("Model config saved as model_config.json")

        return True

    except Exception as e:
        logging.error("Fatal error during training")
        logging.error(str(e))
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'training_dataset.csv'
    train_model(dataset_path)
    logging.info("Training complete.")
    logging.info(f"Output files saved in: {OUTPUT_DIR}")
    sys.exit(0 if train_model(dataset_path) else 1)
    
