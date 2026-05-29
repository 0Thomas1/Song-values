import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Defaults mirror training/train.py
CSV_PATH = "data_generation/kaggle_raw/kaggle_lyrics_sample.csv"
JSONL_PATH = "sythesized_labels/kaggle_synthesized_labels.jsonl"
MODEL_DEFAULT = "models/morale_classifier_v3_random_forest.joblib"

# Keep the same label order used during training
MORAL_LABELS = [
    "Love", "Hate",
    "Joy", "Despair",
    "Peace", "Conflict",
    "Patience", "Impatience",
    "Kindness", "Cruelty",
    "Goodness", "Malice",
    "Faithfulness", "Betrayal",
    "Gentleness", "Harshness",
    "Self_control", "Recklessness"
]


def load_and_merge_data(csv_path: str, jsonl_path: str):
    """Load lyrics CSV and JSONL synthesized labels, returning texts and label matrix.

    This mirrors the logic used in training so evaluation splits match expectations.
    """
    df_lyrics = pd.read_csv(csv_path)
    df_lyrics['song'] = df_lyrics['song'].astype(str)
    df_lyrics = df_lyrics.drop_duplicates(subset='song', keep='first')

    labels_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            labels_data.append(json.loads(line))

    df_labels = pd.DataFrame(labels_data)
    df_labels['song_id'] = df_labels['song_id'].astype(str)
    df_labels = df_labels.drop_duplicates(subset='song_id', keep='first')

    df_master = pd.merge(
        left=df_lyrics,
        right=df_labels,
        left_on='song',
        right_on='song_id',
        how='inner',
        validate='one_to_one'
    )

    df_master = df_master[df_master['lyrics'].notna()].copy()
    df_master['lyrics'] = df_master['lyrics'].astype(str).str.strip()
    df_master = df_master[df_master['lyrics'] != '']

    x_text = df_master['lyrics'].tolist()
    y_matrix = df_master[MORAL_LABELS].astype(int).to_numpy()

    return x_text, y_matrix


def stack_positive_probabilities(proba_list):
    """Given a list of (n_samples, n_classes) arrays, return (n_samples, n_labels)
    of positive-class probabilities (column 1 for binary classifiers).
    """
    # Each element p in proba_list is array shape (n_samples, n_classes)
    pos_cols = [p[:, 1] for p in proba_list]
    return np.vstack(pos_cols).T


def evaluate(model_path: str, csv_path: str, jsonl_path: str, threshold: float):
    # Load data
    print(f"Loading data from {csv_path} and {jsonl_path}...")
    x_text, y = load_and_merge_data(csv_path, jsonl_path)

    # Load model bundle (expects dict with 'vectorizer' and 'classifier')
    model_bundle = joblib.load(model_path)
    vectorizer = model_bundle['vectorizer']
    classifier = model_bundle['classifier']

    # Vectorize everything, then split so we evaluate on the same relative portions
    X_vectors = vectorizer.transform(x_text)

    X_train, x_temp, y_train, y_temp = train_test_split(
        X_vectors, y, test_size=0.3, random_state=42
    )
    X_test, x_val, y_test, y_val = train_test_split(
        x_temp, y_temp, test_size=1 / 3, random_state=42
    )

    print(f"Evaluating with global threshold = {threshold}")

    # Validation set
    if hasattr(classifier, 'predict_proba'):
        # MultiOutputClassifier.predict_proba returns list of arrays
        val_proba_list = classifier.predict_proba(x_val)
        val_pos = stack_positive_probabilities(val_proba_list)
        val_pred = (val_pos >= threshold).astype(int)
    else:
        print("Classifier has no predict_proba; using hard predictions instead.")
        val_pred = classifier.predict(x_val)

    print("\nValidation Report:\n")
    print(classification_report(y_val, val_pred, target_names=MORAL_LABELS, zero_division=0))

    # Test set
    if hasattr(classifier, 'predict_proba'):
        test_proba_list = classifier.predict_proba(X_test)
        test_pos = stack_positive_probabilities(test_proba_list)
        test_pred = (test_pos >= threshold).astype(int)
    else:
        test_pred = classifier.predict(X_test)

    print("\nTest Report:\n")
    print(classification_report(y_test, test_pred, target_names=MORAL_LABELS, zero_division=0))


def main():
    p = argparse.ArgumentParser(description="Evaluate saved morale classifier with a global threshold")
    p.add_argument('model_path', nargs='?', default=MODEL_DEFAULT, help='Path to joblib model bundle')
    p.add_argument('--csv', default=CSV_PATH, help='Path to lyrics CSV')
    p.add_argument('--jsonl', default=JSONL_PATH, help='Path to synthesized JSONL labels')
    p.add_argument('--threshold', type=float, default=0.65, help='Global positive-class threshold')

    args = p.parse_args()

    model_file = Path(args.model_path)
    if not model_file.exists():
        raise SystemExit(f"Model not found: {model_file}")

    evaluate(str(model_file), args.csv, args.jsonl, args.threshold)


if __name__ == '__main__':
    main()
