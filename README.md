# Song Morale Classifier

A machine learning pipeline that analyzes song lyrics and classifies them across 18 distinct moral and emotional themes using semantic embeddings and multi-label classification.

> **🚀 NEW: FastText Upgrade Available!**
>
> We've created an improved pipeline using **FastText** and a larger **Kaggle dataset** for better performance and speed. See [FASTTEXT_UPGRADE_GUIDE.md](FASTTEXT_UPGRADE_GUIDE.md) for details.

## Overview

This repository contains two implementations:

### Current Model (v1)

- **Data**: 3,781 songs from Bruno Mars dataset
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Classifier**: Multi-label Logistic Regression
- **Performance**: F1-Score 0.61 (weighted)
- **Status**: Production ready

### Improved Model (v2) - FastText

- **Data**: 50K+ songs from Kaggle (Genius lyrics)
- **Algorithm**: FastText with multi-label support
- **Benefits**: 10-100x faster, lower memory, better OOV handling
- **Status**: Under development - see upgrade guide
- **Expected**: F1-Score 0.70+ (estimated)

---

## Data Flow

```
all_lyrics.csv (raw lyrics) ──┐
                               ├──> Data Merging & Cleaning ──> training/train.py
synthesized_labels.jsonl ─────┘

train.py pipeline:
  1. Load & merge lyrics with labels
  2. Embed lyrics with sentence-transformers
  3. Split data (80/20 train/test)
  4. Train multi-label classifier
  5. Evaluate & save model
  └─> morale_classifier_v1.joblib (production model)
```

### Input Data

- **`data_generation/lyrics/all_lyrics.csv`**: Contains song titles and their full lyrics
- **`synthesized_labels.jsonl`**: JSONL file where each line contains:
  - `song_id`: Song title (must match CSV)
  - 18 binary moral labels (e.g., `"Love": true/false`, `"Hate": true/false`, etc.)

### Data Processing

1. **Load & Deduplicate**: Remove duplicate songs by title, keeping only the first occurrence
2. **Merge**: Inner join on song title to align lyrics with labels
3. **Clean**: Remove rows with missing or empty lyrics
4. **Result**: 3,781 matched and cleaned songs ready for training

---

## Model Architecture

### Embedding Layer

- **Model**: `all-MiniLM-L6-v2` (384-dimensional semantic vectors)
- **Purpose**: Converts song lyrics into fixed-size semantic representations
- **Processing**: ~42 seconds to embed 3,781 songs in batches

### Classification Layer

- **Algorithm**: Multi-Label Logistic Regression
- **Approach**: OneVsRest with 18 independent binary classifiers
- **Balance**: `class_weight='balanced'` to handle imbalanced labels (e.g., rare "Hate" or "Cruelty" themes)
- **Iterations**: 2,000 max iterations for convergence

### Training/Test Split

- **Train Set**: 80% (3,024 songs)
- **Test Set**: 20% (757 songs)

---

## Performance Results

### Classification Metrics

| Moral Theme      | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Love**         | 0.92      | 0.74   | 0.82     | 600     |
| **Hate**         | 0.29      | 0.62   | 0.39     | 115     |
| **Joy**          | 0.71      | 0.65   | 0.68     | 417     |
| **Despair**      | 0.71      | 0.66   | 0.68     | 446     |
| **Peace**        | 0.48      | 0.54   | 0.51     | 306     |
| **Conflict**     | 0.68      | 0.62   | 0.65     | 456     |
| **Patience**     | 0.44      | 0.51   | 0.47     | 306     |
| **Impatience**   | 0.52      | 0.64   | 0.57     | 286     |
| **Kindness**     | 0.69      | 0.67   | 0.68     | 412     |
| **Cruelty**      | 0.48      | 0.59   | 0.53     | 250     |
| **Goodness**     | 0.85      | 0.64   | 0.73     | 599     |
| **Malice**       | 0.30      | 0.62   | 0.41     | 139     |
| **Faithfulness** | 0.42      | 0.54   | 0.47     | 269     |
| **Betrayal**     | 0.58      | 0.51   | 0.54     | 423     |
| **Gentleness**   | 0.41      | 0.60   | 0.49     | 245     |
| **Harshness**    | 0.48      | 0.68   | 0.56     | 229     |
| **Self_control** | 0.64      | 0.55   | 0.59     | 443     |
| **Recklessness** | 0.42      | 0.63   | 0.50     | 193     |

### Overall Performance

- **Macro Average**: 56% Precision, 61% Recall, 57% F1-Score
- **Weighted Average**: 62% Precision, 62% Recall, 61% F1-Score
- **Micro Average**: 58% Precision, 62% Recall, 60% F1-Score

### Key Insights

✅ **Strong Performers**: Love (0.82), Goodness (0.73), Joy & Despair (~0.68)

- High precision on "Love" (0.92) suggests clear semantic patterns
- Balanced performance on core emotions

⚠️ **Moderate Performers**: Most themes cluster around 0.50-0.65 F1-score

- Adequate for categorization but could benefit from:
  - More labeled training data
  - Fine-tuned embeddings
  - Larger ensemble models

❌ **Challenging Themes**: Hate (0.39), Malice (0.41)

- Low precision but high recall (catches most cases, flags some false positives)
- Likely due to limited training samples (115 & 139 instances)
- `class_weight='balanced'` prevents the model from ignoring these rare labels

---

## Output Artifact

**File**: `morale_classifier_v1.joblib`

- **Type**: Scikit-learn serialized model (MultiOutputClassifier)
- **Format**: Binary joblib format
- **Size**: ~2-5 MB (depends on embeddings cache)
- **Usage**: Load with `joblib.load()` for inference

### Production Inference Example

```python
import joblib
from sentence_transformers import SentenceTransformer

model = joblib.load('morale_classifier_v1.joblib')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

lyrics = "Some song lyrics here..."
embedding = embedder.encode([lyrics])
predictions = model.predict(embedding)  # Returns binary array [0/1] for each theme
```

---

## 18 Moral/Emotional Themes (Fruit of the Spirit Framework)

The labels are organized as opposite pairs:

1. **Love** ↔ **Hate**
2. **Joy** ↔ **Despair**
3. **Peace** ↔ **Conflict**
4. **Patience** ↔ **Impatience**
5. **Kindness** ↔ **Cruelty**
6. **Goodness** ↔ **Malice**
7. **Faithfulness** ↔ **Betrayal**
8. **Gentleness** ↔ **Harshness**
9. **Self_control** ↔ **Recklessness**

---

## Dependencies

install:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── README.md (this file)
├── morale_classifier_v1.joblib          # Trained production model
├── synthesized_labels.jsonl             # Generated labels from LM Studio
├── data_generation/
│   ├── label_kaggle_lyrics.py                # Label generation script
│   ├── download_kaggle.py                     # Kaggle data download script
└── training/
    └── train_fasttext.py                         # Main training pipeline
```

---

## Status

✅ **Training Complete**, model saved, and ready for production use. See `inference_fasttext.py` for example usage.

- 8060 songs processed
- 1610 test samples evaluated
- Weighted F1-Score: 0.6
