# LyricsLens

LyricsLens is a machine learning pipeline that analyzes song lyrics and classifies them across 18 moral and emotional themes. LyricsLens provides a scikit-learn-based training pipeline (TF-IDF + Random Forest) that avoids a PyTorch dependency.

## Model Training

LyricsLens uses a two-stage approach:

1. **Data Generation**: LM Studio or other tooling to synthesize labels for songs
2. **Model Training**: TF-IDF + RandomForest for multi-label classification

---

### Training/Test Split

- **Train Set**: 70% of the data
- **Test Set**: 20% of the data
- **Validation Set**: 10% of the data

---

## Output Artifact

---

## Training Usage

Prepare a Python environment and install dependencies, then run the training script from the repository root.

Install dependencies (recommended):

```bash
# create and activate a virtual environment (optional)
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install pandas numpy joblib scikit-learn
```

Run training:

```bash
python training/train.py
```

Notes:

- Ensure your raw lyrics CSV and synthesized labels are available at the paths expected by `training/train.py` (see the top of that file for `CSV_PATH` and `JSONL_PATH`).
- The training script will create a `models/` directory and save the trained bundle to `models/morale_classifier_v3_random_forest.joblib`.

**File**: `models/morale_classifier_v3_random_forest.joblib`

- **Type**: Scikit-learn serialized bundle (dict with `vectorizer` and `classifier`)
- **Usage**: Load with `joblib.load()` and use `bundle['vectorizer']` and `bundle['classifier']` for inference

---

## Inference Usage

Run inference with the CLI script in `training/inference.py`.

### 1) Predict from inline text

```bash
python training/inference.py models/morale_classifier_v3_random_forest.joblib --text "I love you and I feel peace and joy" --pretty
```

### 2) Predict from a text file

```bash
python training/inference.py models/morale_classifier_v3_random_forest.joblib --text-file lyrics.txt --pretty
```

### 3) Use a custom probability threshold

```bash
python training/inference.py models/morale_classifier_v3_random_forest.joblib --text-file lyrics.txt --threshold 0.65 --pretty
```

Recommended: use `--threshold 0.65` for this model, as it currently gives the best practical results.

The script outputs JSON with:

- `positive_labels`: labels predicted as true
- `all_labels`: all 18 labels with predicted boolean values (and probabilities when available)

---

## Notes

- LyricsLens defaults to a pure scikit-learn pipeline to avoid PyTorch DLL issues on some Windows setups.
- The training script saves the fitted `TfidfVectorizer` together with the `MultiOutputClassifier(RandomForestClassifier)` so that preprocessing is preserved for inference.
- Older model artifacts (if present) include `morale_classifier_v1.joblib` and `models/morale_classifier_v2_sklearn.joblib`.
