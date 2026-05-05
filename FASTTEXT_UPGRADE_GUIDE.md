# FastText Model Upgrade Guide

## Overview

This guide walks through the improved pipeline that uses **FastText** for multi-label classification and a **larger Kaggle dataset** for training. FastText offers:

- **Speed**: 10-100x faster than Sentence-Transformers
- **Efficiency**: Low memory footprint, can handle large datasets
- **Robustness**: Better handling of out-of-vocabulary words via subword information
- **Simplicity**: No need for separate embedding + classification steps

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Download & Process Kaggle Dataset                  │
│         (carlosgdcj/genius-song-lyrics-with-language)      │
│         Output: kaggle_lyrics.csv (~50K+ songs)            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Step 2: Generate Labels via LM Studio                       │
│         Process lyrics → Extract 18 moral themes            │
│         Output: kaggle_synthesized_labels.jsonl             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Step 3: Train FastText Multi-Label Classifier              │
│         FastText can handle multiple labels per sample      │
│         Output: morale_classifier_fasttext_v2.bin           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Step 4: Inference & Evaluation                             │
│         Compare with old model, deploy to production        │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

### 2. Configure Kaggle Credentials

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New Token" (downloads `kaggle.json`)
4. Place in `~/.kaggle/kaggle.json`

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Install New Dependencies

```bash
pip install -r requirements.txt
```

Key new packages:

- `fasttext` - Fast text classification
- `kaggle` - Download datasets
- Updated `pydantic`, `scikit-learn`, `aiofiles`

---

## Step-by-Step Execution

### Step 1: Download Kaggle Dataset (10K Sample - NOT Full 9.7GB)

```bash
cd /Users/funbandit/Desktop/GITHUB/Song\ values
python data_generation/download_kaggle.py
```

**Output:**

- Downloads Genius lyrics dataset
- **Samples only 10,000 songs** (not full dataset)
- Extracts and cleans
- Saves to `data_generation/kaggle_lyrics_sample.csv` (~50MB)

**Expected time:** 5-15 minutes
**Space saved:** Downloads full 9.7GB but samples it, keeping only ~50MB of useful data ✅

---

### Step 2: Generate Labels (TWO OPTIONS)

You now have two paths. Choose one:

#### Option A: FastText Transfer Learning (RECOMMENDED ✅ - 5 minutes)

Use your existing 3,781 labeled songs as a "teacher" to label the new 10,000 Kaggle songs.

```bash
python data_generation/synthesize_labels_fasttext.py
```

**What it does:**

1. Trains FastText on your existing 3,781 labeled songs
2. Uses that model to predict labels for 10,000 new songs
3. Very fast & free, no external services needed

**Output:** `kaggle_synthesized_labels_fasttext.jsonl`
**Time:** ~2-5 minutes ⚡

**Advantages:**

- ⚡ 100x faster than LM Studio
- 🔒 Deterministic and consistent
- 💰 Free, no API calls
- 📴 Works offline
- 📊 Already tuned to your label style

**See also:** [FASTTEXT_LABEL_SYNTHESIS.md](FASTTEXT_LABEL_SYNTHESIS.md) for detailed guide

---

#### Option B: LM Studio Generation (Original - 2-6 hours)

Use LM Studio local LLM to analyze each song and extract moral themes.

```bash
python data_generation/label_kaggle_lyrics.py
```

**What it does:**

1. Sends each song to LM Studio
2. Gets nuanced LLM analysis
3. Returns detailed labels

**Output:** `kaggle_synthesized_labels.jsonl`
**Time:** 2-6 hours (depends on LM Studio throughput)

**Requirements:** LM Studio must be running (`localhost:1234`)

⚠️ **Note:** This is slower but may capture more nuanced interpretations

---

### Step 3: Train Final Model

Choose based on which labeling method you used above:

#### If you used FastText labels:

```bash
python training/train_combined_fasttext.py
```

**Output:**

- `models/morale_classifier_combined_v2.bin` - Final model trained on 13,781 songs
- `models/morale_classifier_combined_v2_metadata.json` - Model metadata

**Time:** 5-15 minutes

#### If you used LM Studio labels:

```bash
python training/train_fasttext.py
```

**Output:**

- `models/morale_classifier_fasttext_v2.bin`
- `models/morale_classifier_fasttext_v2_metadata.json`

**Time:** 5-15 minutes

---

### Step 4: Inference & Evaluation

#### Option A: Interactive Mode

```bash
python inference_fasttext.py
```

Example output:

```
Predictions (threshold=0.3):
  Love           0.925 ████████████████████
  Joy            0.891 ███████████████████
  Goodness       0.756 ███████████
  Peace          0.634 ██████
  Kindness       0.521 ████
```

#### Option B: Programmatic Use

```python
from inference_fasttext import predict_lyrics

lyrics = "Your song lyrics here..."
result = predict_lyrics(lyrics, threshold=0.5)

print(result['active_themes'])  # ['Love', 'Joy', 'Peace']
print(result['themes'])         # {'Love': 0.92, 'Hate': 0.15, ...}
```

#### Option C: Compare Models

Compare FastText against the old Sentence-Transformers model:

```bash
python training/compare_models.py
```

This will:

- Load both models
- Run predictions on 100 test samples
- Generate classification reports for each
- Show performance differences

---

## FastText vs Sentence-Transformers

| Aspect            | FastText      | Sentence-Transformers       |
| ----------------- | ------------- | --------------------------- |
| **Speed**         | ⚡ Very Fast  | 🐢 Slower                   |
| **Memory**        | 📦 ~50MB      | 📦📦 ~500MB+                |
| **Training Time** | ⏱️ 5-15 min   | ⏱️ 30+ min                  |
| **Embeddings**    | 100-dim       | 384-dim                     |
| **OOV Handling**  | ✓ Subword     | ✗ None                      |
| **Multi-label**   | ✓ Native      | ✓ Via MultiOutputClassifier |
| **Production**    | ✓ Binary file | ✓ Joblib serialization      |

---

## Performance Expectations

### With ~50K Kaggle Songs:

**Estimated Improvements:**

- ✅ Better coverage of diverse lyrics styles
- ✅ More robust to rare moral themes
- ✅ Faster inference (10-100x)
- ✅ Smaller model file size
- ⚠️ May require rebalancing for rare classes

### Benchmark (on 100 samples):

```
Old Model (3,781 songs):
  Weighted F1: 0.61
  Hamming Loss: 0.38

Expected FastText (50K+ songs):
  Weighted F1: 0.70+ (estimate)
  Hamming Loss: 0.30-0.35
```

---

## Troubleshooting

### Issue: Kaggle API not found

**Solution:**

```bash
pip install kaggle
# Check credentials:
ls -la ~/.kaggle/kaggle.json
```

### Issue: LM Studio connection error

**Solution:**

- Make sure LM Studio is running (`localhost:1234`)
- Check firewall settings
- Verify LM Studio model is loaded

### Issue: Out of memory during FastText training

**Solution:**

```python
# In train_fasttext.py, reduce:
epoch=15  # was 25
dim=50    # was 100
```

### Issue: Model file too large

**Solution:**
FastText models are typically 50-200MB. If larger:

```python
# Reduce vocabulary during training:
model.save_model(path, epoch=10)  # Fewer epochs
```

---

## Production Deployment

### 1. Save Model Path

```python
MODEL_PATH = "models/morale_classifier_fasttext_v2.bin"
```

### 2. Load in Production

```python
import fasttext

model = fasttext.load_model(MODEL_PATH)
predictions, scores = model.predict(text, k=18)
```

### 3. Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY models/ ./models/
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY inference_fasttext.py .
CMD ["python", "inference_fasttext.py"]
```

---

## File Structure

```
Song values/
├── data_generation/
│   ├── download_kaggle.py          # NEW: Download Kaggle data
│   ├── label_kaggle_lyrics.py       # NEW: Generate labels via LM Studio
│   ├── kaggle_raw/                  # NEW: Downloaded dataset
│   ├── kaggle_lyrics.csv            # NEW: Processed Kaggle lyrics
│   └── ...
├── training/
│   ├── train_fasttext.py            # NEW: FastText training
│   ├── compare_models.py            # NEW: Model comparison
│   ├── train.py                     # OLD: Sentence-Transformers pipeline
│   └── ...
├── models/                          # NEW: Model storage
│   ├── morale_classifier_fasttext_v2.bin
│   └── morale_classifier_fasttext_v2_metadata.json
├── inference_fasttext.py            # NEW: FastText inference
├── requirements.txt                 # UPDATED: New dependencies
└── ...
```

---

## Next Steps

1. **Download & Label**: Run steps 1-2 (allow 6-24 hours for labeling)
2. **Train**: Run step 3 (15 minutes)
3. **Evaluate**: Run step 5 to compare with old model
4. **Deploy**: Use `inference_fasttext.py` for predictions
5. **Iterate**: Collect user feedback, retrain with corrections

---

## References

- FastText Documentation: https://fasttext.cc/
- Kaggle API: https://github.com/Kaggle/kaggle-api
- Multi-label Classification: https://scikit-learn.org/stable/modules/multiclass.html
