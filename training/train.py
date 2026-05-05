import pandas as pd
import numpy as np
import json
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================================
# 1. Configuration & Taxonomy
# ==========================================
CSV_PATH = "data_generation/lyrics/all_lyrics.csv"
JSONL_PATH = "synthesized_labels.jsonl"
MODEL_SAVE_PATH = "morale_classifier_v1.joblib"

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
    X_text = df_master['lyrics'].tolist()
    y_matrix = df_master[MORAL_LABELS].astype(int).to_numpy()

    return X_text, y_matrix

# ==========================================
# 3. Model Training Pipeline
# ==========================================
def train_model():
    # Step A: Get the aligned data
    X_text, y = load_and_merge_data()

    # Step B: Embed the text locally
    # We pass the entire list to .encode() so it batches them automatically (much faster)
    print("\nLoading local embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Embedding lyrics into semantic vectors (This may take a minute or two)...")
    X_vectors = embedder.encode(X_text, show_progress_bar=True)

    # Step C: Split the data into Training (80%) and Testing (20%)
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42
    )

    # Step D: Initialize and Train the Classifier
    print("Training the Multi-Label Logistic Regression Classifier...")
    # class_weight='balanced' is critical here so rare tags (like Cruelty) aren't ignored
    base_estimator = LogisticRegression(max_iter=2000, class_weight='balanced')
    classifier = MultiOutputClassifier(base_estimator)
    
    classifier.fit(X_train, y_train)

    # Step E: Evaluate the Model
    print("\nEvaluating Model Accuracy on Test Set:")
    predictions = classifier.predict(X_test)
    
    # zero_division=0 prevents warnings if a class was completely missed in the test set split
    report = classification_report(
        y_test, predictions, target_names=MORAL_LABELS, zero_division=0
    )
    print(report)

    # Step F: Save the compiled model
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    joblib.dump(classifier, MODEL_SAVE_PATH)
    print("Training pipeline complete! You are ready for production inference.")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    train_model()