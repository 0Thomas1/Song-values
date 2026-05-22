# Song Morale Classifier

A machine learning pipeline that analyzes song lyrics and classifies them across 18 moral and emotional themes. This repository now provides a scikit-learn-based training pipeline (TF-IDF + Random Forest) that avoids a PyTorch dependency.

## Model Training

This project now uses a two-stage approach:

1. **Data Generation**: LM Studio or other tooling to synthesize labels for songs
2. **Model Training**: TF-IDF + RandomForest for multi-label classification

---

### Training/Test Split

- **Train Set**: 70% of the data
- **Test Set**: 20% of the data
- **Validation Set**: 10% of the data

---

## Output Artifact

**File**: `models/morale_classifier_v3_random_forest.joblib`

- **Type**: Scikit-learn serialized bundle (dict with `vectorizer` and `classifier`)
- **Usage**: Load with `joblib.load()` and use `bundle['vectorizer']` and `bundle['classifier']` for inference

---

## Notes

- The repository now defaults to a pure scikit-learn pipeline to avoid PyTorch DLL issues on some Windows setups.
- The training script saves the fitted `TfidfVectorizer` together with the `MultiOutputClassifier(RandomForestClassifier)` so that preprocessing is preserved for inference.
- Older model artifacts (if present) include `morale_classifier_v1.joblib` and `models/morale_classifier_v2_sklearn.joblib`.


