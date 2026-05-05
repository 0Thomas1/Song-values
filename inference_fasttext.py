"""
Inference script for FastText moral classifier.
Use this to classify new songs or compare with old model.
"""

import fasttext
import json
from pathlib import Path

MODEL_PATH = "models/morale_classifier_fasttext_v2.bin"
METADATA_PATH = "models/morale_classifier_fasttext_v2_metadata.json"

def load_model():
    """Load FastText model."""
    model = fasttext.load_model(MODEL_PATH)
    
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    
    return model, metadata

def preprocess_text(text):
    """Preprocess text for FastText."""
    text = text.lower()
    text = " ".join(text.split())
    return text

def predict_lyrics(lyrics, threshold=0.5):
    """
    Predict moral themes for given lyrics.
    
    Args:
        lyrics: Song lyrics text
        threshold: Confidence threshold for predictions (0-1)
    
    Returns:
        dict: {
            'themes': {theme_name: confidence, ...},
            'active_themes': [theme_names with confidence >= threshold],
            'raw_prediction': FastText output
        }
    """
    model, metadata = load_model()
    labels = metadata['labels']
    
    # Preprocess
    processed = preprocess_text(lyrics)
    
    # Predict all labels with confidence
    predictions, scores = model.predict(processed, k=len(labels))
    
    # Clean up label names
    theme_scores = {}
    for pred_label, score in zip(predictions, scores):
        theme_name = pred_label.replace("__label__", "")
        theme_scores[theme_name] = float(score)
    
    # Filter by threshold
    active_themes = [theme for theme, score in theme_scores.items() if score >= threshold]
    
    return {
        'themes': theme_scores,
        'active_themes': active_themes,
        'threshold': threshold,
        'num_themes': len(active_themes)
    }

def batch_predict(lyrics_list, threshold=0.5):
    """Predict for multiple lyrics."""
    results = []
    for lyrics in lyrics_list:
        result = predict_lyrics(lyrics, threshold)
        results.append(result)
    return results

def main():
    """Interactive inference example."""
    model, metadata = load_model()
    
    print("="*60)
    print("FastText Moral Theme Classifier - Interactive Mode")
    print("="*60)
    print(f"\nLoaded model with {len(metadata['labels'])} themes")
    print(f"Trained on {metadata['dataset_size']} songs\n")
    
    # Example usage
    sample_lyric = """
    I see the love in your eyes, the kindness in your heart
    Together we can find the peace that holds us from the start
    No hate or cruelty can tear us apart
    Our faithfulness is written in every part
    """
    
    print("Example prediction:")
    print("-" * 60)
    print(f"Lyrics: {sample_lyric[:100]}...\n")
    
    result = predict_lyrics(sample_lyric, threshold=0.3)
    
    print("Predictions (threshold=0.3):")
    for theme, score in sorted(result['themes'].items(), key=lambda x: x[1], reverse=True)[:10]:
        bar = "█" * int(score * 20)
        print(f"  {theme:15} {score:.3f} {bar}")
    
    print(f"\nActive themes: {', '.join(result['active_themes']) if result['active_themes'] else 'None'}")

if __name__ == "__main__":
    main()
