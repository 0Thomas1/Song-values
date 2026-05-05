"""
Compare performance between old model (Sentence-Transformers + Logistic Regression)
and new model (FastText) on the same test set.
"""

import pandas as pd
import numpy as np
import json
import joblib
import fasttext
from sklearn.metrics import classification_report, f1_score, hamming_loss

# Configuration
CSV_PATH = "data_generation/kaggle_lyrics.csv"
JSONL_PATH = "kaggle_synthesized_labels.jsonl"

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

def load_test_data():
    """Load and prepare test data."""
    print("Loading data...")
    
    df_lyrics = pd.read_csv(CSV_PATH)
    df_lyrics['song'] = df_lyrics['song'].astype(str)
    
    labels_data = []
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            labels_data.append(json.loads(line))
    
    df_labels = pd.DataFrame(labels_data)
    df_labels['song_id'] = df_labels['song_id'].astype(str)
    
    # Merge
    df_master = pd.merge(
        df_lyrics,
        df_labels,
        left_on='song',
        right_on='song_id',
        how='inner'
    )
    
    df_master = df_master[df_master['lyrics'].notna()]
    df_master['lyrics'] = df_master['lyrics'].astype(str).str.strip()
    df_master = df_master[df_master['lyrics'] != '']
    
    X_text = df_master['lyrics'].tolist()
    y_true = df_master[MORAL_LABELS].astype(int).to_numpy()
    
    return X_text, y_true

def predict_fasttext(model, texts):
    """Get predictions from FastText model."""
    predictions = []
    for text in texts:
        text_lower = text.lower()
        text_clean = " ".join(text_lower.split())
        
        # Get predictions for all labels
        pred_labels, scores = model.predict(text_clean, k=len(MORAL_LABELS))
        
        # Convert to binary vector
        pred_vector = np.array([1 if label.replace("__label__", "") in MORAL_LABELS else 0 
                               for label in pred_labels[:len(MORAL_LABELS)]])
        predictions.append(pred_vector)
    
    return np.array(predictions)

def main():
    print("="*70)
    print("Model Comparison: FastText vs Sentence-Transformers + Logistic Regression")
    print("="*70)
    
    # Load test data
    X_test, y_test = load_test_data()
    print(f"\n✓ Loaded {len(X_test)} test samples")
    
    # Use first 100 samples for quick comparison (or adjust as needed)
    sample_size = min(100, len(X_test))
    X_sample = X_test[:sample_size]
    y_sample = y_test[:sample_size]
    
    print(f"Using {sample_size} samples for comparison\n")
    
    try:
        # Load FastText model
        print("Loading FastText model...")
        ft_model = fasttext.load_model("models/morale_classifier_fasttext_v2.bin")
        ft_predictions = predict_fasttext(ft_model, X_sample)
        print("✓ FastText predictions complete")
        
        # FastText metrics
        print("\n" + "="*70)
        print("FastText Model Performance")
        print("="*70)
        print(classification_report(y_sample, ft_predictions, target_names=MORAL_LABELS, zero_division=0))
        ft_f1 = f1_score(y_sample, ft_predictions, average='weighted', zero_division=0)
        ft_hamming = hamming_loss(y_sample, ft_predictions)
        print(f"Weighted F1-Score: {ft_f1:.4f}")
        print(f"Hamming Loss: {ft_hamming:.4f}")
        
    except Exception as e:
        print(f"✗ FastText model not found: {e}")
        ft_f1 = None
        ft_hamming = None
    
    try:
        # Load old model
        print("\n" + "="*70)
        print("Old Model (Sentence-Transformers + Logistic Regression)")
        print("="*70)
        
        from sentence_transformers import SentenceTransformer
        old_model = joblib.load("morale_classifier_v1.joblib")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Embedding samples...")
        X_vectors = embedder.encode(X_sample, show_progress_bar=True)
        old_predictions = old_model.predict(X_vectors)
        print("✓ Old model predictions complete")
        
        print("\n" + "="*70)
        print("Old Model Performance")
        print("="*70)
        print(classification_report(y_sample, old_predictions, target_names=MORAL_LABELS, zero_division=0))
        old_f1 = f1_score(y_sample, old_predictions, average='weighted', zero_division=0)
        old_hamming = hamming_loss(y_sample, old_predictions)
        print(f"Weighted F1-Score: {old_f1:.4f}")
        print(f"Hamming Loss: {old_hamming:.4f}")
        
    except Exception as e:
        print(f"✗ Old model not found: {e}")
        old_f1 = None
        old_hamming = None
    
    # Summary
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    
    if ft_f1 and old_f1:
        print(f"{'Metric':<25} {'FastText':<20} {'Old Model':<20} {'Difference':<15}")
        print("-" * 80)
        
        f1_diff = ft_f1 - old_f1
        hamming_diff = ft_hamming - old_hamming
        
        print(f"{'F1-Score (weighted)':<25} {ft_f1:<20.4f} {old_f1:<20.4f} {f1_diff:+.4f}")
        print(f"{'Hamming Loss':<25} {ft_hamming:<20.4f} {old_hamming:<20.4f} {hamming_diff:+.4f}")
        
        if ft_f1 > old_f1:
            improvement = ((ft_f1 - old_f1) / old_f1) * 100
            print(f"\n✓ FastText is {improvement:.1f}% better on F1-Score")
        else:
            degradation = ((old_f1 - ft_f1) / old_f1) * 100
            print(f"\n⚠ Old model is {degradation:.1f}% better on F1-Score")
    else:
        print("Could not complete comparison - one or both models missing")

if __name__ == "__main__":
    main()
