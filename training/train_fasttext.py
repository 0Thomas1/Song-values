"""
Train FastText classifiers for multi-label moral theme classification.
FastText is faster and more efficient than sentence-transformers + logistic regression.
"""

import pandas as pd
import numpy as np
import json
import joblib
import tempfile
from pathlib import Path
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss

# ==========================================
# Configuration
# ==========================================
CSV_PATH_CANDIDATES = [
    "data_generation/kaggle_lyrics_sample.csv",
    "data_generation/kaggle_lyrics.csv",
    "data_generation/lyrics/all_lyrics.csv",
]
JSONL_PATH = "kaggle_synthesized_labels.jsonl"
MODEL_SAVE_DIR = "models/"
MODEL_NAME = "morale_classifier_fasttext_v2"

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
# Data Preparation
# ==========================================
def load_and_merge_data():
    """Load lyrics and labels, merge on song ID."""
    csv_path = next((path for path in CSV_PATH_CANDIDATES if Path(path).exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "No lyrics CSV found. Expected one of: " + ", ".join(CSV_PATH_CANDIDATES)
        )

    print(f"Loading {csv_path} and {JSONL_PATH}...")

    df_lyrics = pd.read_csv(csv_path)
    df_lyrics['song'] = df_lyrics['song'].astype(str)
    df_lyrics = df_lyrics.drop_duplicates(subset='song', keep='first')
    
    labels_data = []
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            labels_data.append(json.loads(line))
    
    df_labels = pd.DataFrame(labels_data)
    df_labels['song_id'] = df_labels['song_id'].astype(str)
    df_labels = df_labels.drop_duplicates(subset='song_id', keep='first')
    
    # Merge on song title
    df_master = pd.merge(
        left=df_lyrics,
        right=df_labels,
        left_on='song',
        right_on='song_id',
        how='inner',
        validate='one_to_one'
    )
    
    # Clean lyrics
    df_master = df_master[df_master['lyrics'].notna()].copy()
    df_master['lyrics'] = df_master['lyrics'].astype(str).str.strip()
    df_master = df_master[df_master['lyrics'] != '']
    
    print(f"✓ Successfully matched {len(df_master)} songs")
    
    # Extract texts and labels
    X_text = df_master['lyrics'].tolist()
    y_matrix = df_master[MORAL_LABELS].astype(int).to_numpy()
    
    return X_text, y_matrix, df_master

def preprocess_text_for_fasttext(text):
    """Preprocess text for FastText: clean and normalize."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    # FastText treats special characters, so keep basic punctuation
    return text

def format_fasttext_training_data(texts, labels, label_names):
    """Format data for FastText multi-label training."""
    formatted_data = []
    for text, label_vector in zip(texts, labels):
        processed_text = preprocess_text_for_fasttext(text)
        # FastText expects labels as __label__LABEL_NAME
        active_labels = [f"__label__{label_names[i]}" for i, val in enumerate(label_vector) if val == 1]
        if active_labels:  # Only keep samples with at least one label
            formatted_data.append(" ".join(active_labels) + " " + processed_text)
    
    return formatted_data


def predict_binary_vector(model, text, label_names, threshold=0.5):
    """Predict a binary label vector using score thresholding."""
    pred_labels, pred_scores = model.predict(text, k=len(label_names), threshold=0.0)
    score_by_label = {
        label.replace("__label__", ""): float(score)
        for label, score in zip(pred_labels, pred_scores)
    }
    return np.array([
        1 if score_by_label.get(label, 0.0) >= threshold else 0
        for label in label_names
    ])

# ==========================================
# Model Training
# ==========================================
def train_fasttext_model(train_texts, test_texts, train_labels, test_labels):
    """Train FastText multi-label classifier."""
    print("\n" + "="*50)
    print("Training FastText Multi-Label Classifier")
    print("="*50)
    
    # Format training data
    print("Formatting training data...")
    formatted_train = format_fasttext_training_data(train_texts, train_labels, MORAL_LABELS)
    formatted_test = format_fasttext_training_data(test_texts, test_labels, MORAL_LABELS)
    
    print(f"  Train samples: {len(formatted_train)}")
    print(f"  Test samples: {len(formatted_test)}")
    
    # Write to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        train_file = f.name
        f.write("\n".join(formatted_train))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        test_file = f.name
        f.write("\n".join(formatted_test))
    
    try:
        # Train FastText model
        print("\nTraining model...")
        model = fasttext.train_supervised(
            input=train_file,
            epoch=30,  # More epochs for multi-label
            lr=0.5,
            wordNgrams=3,
            minn=3,  # Subword information
            maxn=6,
            dim=100,  # Embedding dimension
            loss='ova',  # One-vs-all for multi-label classification
            thread=4,
            verbose=2
        )
        
        print("\n✓ Model trained!")
        
        # Evaluate on test set
        print("\nEvaluating on Test Set:")
        print("-" * 50)
        
        # FastText evaluation
        test_acc, test_prec, test_rec = model.test(test_file)
        print(f"FastText Metrics:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall: {test_rec:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print("-" * 50)
        
        # Get predictions for detailed metrics
        prediction_threshold = 0.5
        predictions = []
        ground_truth = []
        
        for i, text in enumerate(test_texts):
            processed_text = preprocess_text_for_fasttext(text)
            pred_vector = predict_binary_vector(
                model,
                processed_text,
                MORAL_LABELS,
                threshold=prediction_threshold,
            )
            predictions.append(pred_vector)
            ground_truth.append(test_labels[i])
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Classification report for each label
        report = classification_report(
            ground_truth,
            predictions,
            target_names=MORAL_LABELS,
            zero_division=0
        )
        print(report)
        
        # Multi-label metrics
        print("\nMulti-Label Metrics:")
        print(f"  Threshold: {prediction_threshold:.2f}")
        print(f"  Hamming Loss: {hamming_loss(ground_truth, predictions):.4f}")
        
        return model
        
    finally:
        # Clean up temporary files
        Path(train_file).unlink()
        Path(test_file).unlink()

# ==========================================
# Main Training Pipeline
# ==========================================
def main():
    # Create models directory
    Path(MODEL_SAVE_DIR).mkdir(exist_ok=True)
    
    # Load and prepare data
    X_text, y_matrix, df_master = load_and_merge_data()
    
    # Split data
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_matrix, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {len(X_train)} songs")
    print(f"  Test: {len(X_test)} songs")
    
    # Train model
    model = train_fasttext_model(X_train, X_test, y_train, y_test)
    
    # Save model
    model_path = f"{MODEL_SAVE_DIR}{MODEL_NAME}.bin"
    model.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "FastText",
        "labels": MORAL_LABELS,
        "dataset_size": len(df_master),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    metadata_path = f"{MODEL_SAVE_DIR}{MODEL_NAME}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_path}")
    print("\n" + "="*50)
    print("Training pipeline complete! Ready for production inference.")
    print("="*50)

if __name__ == "__main__":
    main()
