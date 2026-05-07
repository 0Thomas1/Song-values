# Song Morale Classifier

This repository trains and serves a multi-label FastText classifier that tags song lyrics with 18 moral and emotional themes.

## What This Pipeline Does

The training flow in this project:

1. Loads lyrics from a CSV file.
2. Loads synthesized labels from a JSONL file.
3. Merges both sources on song title.
4. Cleans and normalizes lyric text.
5. Trains a FastText supervised model with one-vs-all loss for multi-label classification.
6. Evaluates predictions with scikit-learn metrics.
7. Saves the model and metadata for inference.

The pipeline is implemented in training/train_fasttext.py.

## Input Data

Training expects:

- One lyrics CSV with song title and lyrics columns.
- One JSONL labels file where each line contains:
  - song_id matching the song title.
  - The 18 binary label fields listed below.

### CSV Discovery Order

The training script automatically picks the first existing CSV from this list:

1. data_generation/kaggle_lyrics_sample.csv
2. data_generation/kaggle_lyrics.csv
3. data_generation/lyrics/all_lyrics.csv

### Labels File

- kaggle_synthesized_labels.jsonl

## Label Set (18 Themes)

1. Love
2. Hate
3. Joy
4. Despair
5. Peace
6. Conflict
7. Patience
8. Impatience
9. Kindness
10. Cruelty
11. Goodness
12. Malice
13. Faithfulness
14. Betrayal
15. Gentleness
16. Harshness
17. Self_control
18. Recklessness

## Training Pipeline Details

### 1) Load and Merge

- Reads lyrics and labels into pandas DataFrames.
- Removes duplicates by song name and song_id.
- Performs an inner merge on song and song_id with one-to-one validation.

### 2) Clean Text

- Drops rows with missing or empty lyrics.
- Lowercases text.
- Collapses repeated whitespace.

### 3) Convert to FastText Format

Each training line is converted to:

**label**LabelA **label**LabelB normalized lyric text...

Only samples with at least one active label are kept.

### 4) Split and Train

- Train/test split: 80/20, random_state=42.
- FastText training settings:
  - loss = ova
  - dim = 100
  - epoch = 30
  - lr = 0.5
  - wordNgrams = 3
  - minn = 3
  - maxn = 6
  - thread = 4

### 5) Evaluate

- FastText built-in test metrics.
- Per-label classification report.
- Hamming loss.
- Binary predictions are created by thresholding label scores (default 0.5).

### 6) Save Artifacts

Saved under models:

- morale_classifier_fasttext_v2.bin
- morale_classifier_fasttext_v2_metadata.json

Metadata includes model type, labels, dataset size, and train/test sample counts.

## Run Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python training/train_fasttext.py
```

## Run Inference

Use inference_fasttext.py for interactive prediction.

```bash
python inference_fasttext.py
```

The inference script:

- Loads the FastText model and metadata.
- Predicts all label scores for input lyrics.
- Returns active themes above a configurable threshold.

## Compare FastText vs Legacy Model

To compare FastText against the older sentence-transformers plus logistic regression model:

```bash
python training/compare_models.py
```

This script reports weighted F1 and hamming loss for available models.

## Project Structure

```text
.
├── README.md
├── inference_fasttext.py
├── kaggle_synthesized_labels.jsonl
├── morale_classifier_v1.joblib
├── requirements.txt
├── data_generation/
│   ├── download_kaggle.py
│   ├── kaggle_lyrics_sample.csv
│   └── label_kaggle_lyrics.py
├── models/
│   └── morale_classifier_fasttext_v2_metadata.json
└── training/
    ├── compare_models.py
    └── train_fasttext.py
```

## Notes

- This README describes the current FastText training pipeline.
- The legacy v1 model file may still exist for backward compatibility and benchmarking.
