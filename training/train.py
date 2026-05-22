import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================================
# 1. Configuration & Taxonomy
# ==========================================
CSV_PATH = "data_generation/kaggle_raw/kaggle_lyrics_sample.csv"
JSONL_PATH = "sythesized_labels\kaggle_synthesized_labels.jsonl"
MODEL_SAVE_PATH = "models/morale_classifier_v3_random_forest.joblib"

# The exact order of these labels dictates the matrix shape. Do not change this order.
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

# ==========================================
# 2. Data Preparation
# ==========================================
def load_and_merge_data() -> tuple[list[str], np.ndarray]:
    """Loads raw lyrics and synthesized labels, merging them perfectly."""
    print(f"Loading {CSV_PATH} and {JSONL_PATH}...")
    
    df_lyrics = pd.read_csv(CSV_PATH)
    df_lyrics['song'] = df_lyrics['song'].astype(str)
    df_lyrics = df_lyrics.drop_duplicates(subset='song', keep='first')

    labels_data = []
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            labels_data.append(json.loads(line))
            
    df_labels = pd.DataFrame(labels_data)
    df_labels['song_id'] = df_labels['song_id'].astype(str)
    df_labels = df_labels.drop_duplicates(subset='song_id', keep='first')

    # Merge on song title to ensure perfect alignment with synthesized labels
    df_master = pd.merge(
        left=df_lyrics, 
        right=df_labels, 
        left_on='song', 
        right_on='song_id', 
        how='inner',
        validate='one_to_one'  # This will raise an error if there are mismatches
    )

    # Drop rows without usable lyrics before embedding.
    df_master = df_master[df_master['lyrics'].notna()].copy()
    df_master['lyrics'] = df_master['lyrics'].astype(str).str.strip()
    df_master = df_master[df_master['lyrics'] != '']
    
    print(f"Successfully matched {len(df_master)} songs after removing duplicate song titles and empty lyrics.")

    # Extract X (Text) and y (Matrix)
    x_text = df_master['lyrics'].tolist()
    y_matrix = df_master[MORAL_LABELS].astype(int).to_numpy()

    return x_text, y_matrix

# ==========================================
# 3. Model Training Pipeline
# ==========================================
def train_model():
    # Step A: Get the aligned data
    x_text, y = load_and_merge_data()

    # Step B: Convert lyrics into TF-IDF vectors.
    print("\nVectorizing lyrics with TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )
    x_vectors = vectorizer.fit_transform(x_text)

    # Step C: Split data into Training (70%), Testing (20%), and Validation (10%)
    print("\nSplitting data into train/test/validation sets (70/20/10)...")
    X_train, x_temp, y_train, y_temp = train_test_split(
        x_vectors, y, test_size=0.3, random_state=42
    )
    # Split the remaining 30% into test (20% overall) and validation (10% overall).
    X_test, x_val, y_test, y_val = train_test_split(
        x_temp, y_temp, test_size=1 / 3, random_state=42
    )

    print(
        f"Split sizes -> train: {len(y_train)}, test: {len(y_test)}, validation: {len(y_val)}"
    )

    # Step D: Initialize and Train the Classifier
    print("Training the Multi-Label Random Forest Classifier...")
    base_estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample"
    )
    classifier = MultiOutputClassifier(base_estimator)
    
    classifier.fit(X_train, y_train)

    # Step E: Evaluate the Model
    print("\nEvaluating Model Accuracy on Validation Set:")
    val_predictions = classifier.predict(x_val)

    # zero_division=0 prevents warnings if a class was completely missed in a split
    val_report = classification_report(
        y_val, val_predictions, target_names=MORAL_LABELS, zero_division=0
    )
    print(val_report)

    print("\nEvaluating Model Accuracy on Test Set:")
    test_predictions = classifier.predict(X_test)
    test_report = classification_report(
        y_test, test_predictions, target_names=MORAL_LABELS, zero_division=0
    )
    print(test_report)

    # Step F: Save the trained bundle (vectorizer + classifier)
    model_path = Path(MODEL_SAVE_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model bundle to {model_path}...")
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_path)
    print("Training pipeline complete! You are ready for production inference.")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    train_model()