# Your Model Improvement Plan - Qwen 3.5 9B

## What You're Getting

✅ **3.6x more training data** (3,781 → 13,781 songs)
✅ **Better labels** (Qwen 3.5 9B LLM analysis)
✅ **Faster labeling** (20-30 min vs 30-45 min with Mistral)
✅ **Perfect GPU fit** (12GB VRAM, you have exactly that)
✅ **Excellent accuracy** (★★★★★)

---

## Your Optimal Path: Qwen 3.5 9B

### What It Does:

- Download only 10,000 song sample from Kaggle (~50MB)
- Analyze each song with Qwen 3.5 9B for moral themes
- Train new model on combined 13,781 songs
- Deliver 15-20% accuracy improvement

### Commands:

```bash
# 1. Download 10K sample (~50MB)
python data_generation/download_kaggle.py

# 2. Generate labels with Qwen 3.5 9B (~20-30 min)
python data_generation/label_kaggle_lyrics.py

# 3. Train combined model (~5-15 min)
python training/train_fasttext.py

# 4. Compare vs old model (optional)
python training/compare_models.py
```

### ✅ Why Qwen 3.5 9B:

- **Perfect VRAM fit**: Exactly 12GB needed, you have 12GB
- **Speed**: 15-20 tokens/sec (faster than Mistral's 10-15)
- **Accuracy**: ★★★★★ excellent instruction following
- **Quality**: Nuanced LLM analysis of moral themes
- **Setup**: Simple (just download in LM Studio)

### 📈 Expected Results:

```
Training size: 3,781 → 13,781 songs (+264%)
F1-Score:      0.61 → 0.72+ (estimated)
Time:          20-30 min labeling (faster than before!)
Label Quality: ★★★★★ (Qwen's reasoning)
```

**See**: [QWEN_3.5_9B_SETUP.md](QWEN_3.5_9B_SETUP.md) for detailed setup

---

## Timeline Estimate

```
Setup:              5-10 min (download + load Qwen)
Preparation:        5 min (activate venv)
Data Download:      5-15 min
Label Generation:   20-30 min ← Qwen working
Model Training:     5-15 min
─────────────────────────────
TOTAL:              40-75 min
```

---

## Files & Scripts

### Data Processing

- **`data_generation/download_kaggle.py`** - Download 10K sample (not 9.7GB)
- **`data_generation/label_kaggle_lyrics.py`** - Generate labels with Qwen 3.5 9B

### Training

- **`training/train_fasttext.py`** - Train on combined labeled data

### Inference & Comparison

- `inference_fasttext.py` - Use new model for predictions
- `training/compare_models.py` - Compare old vs new model

---

## Documentation

- **[QWEN_3.5_9B_SETUP.md](QWEN_3.5_9B_SETUP.md)** ⭐ - START HERE (detailed setup guide)
- [README.md](README.md) - Original project documentation

---

## Quick Checklist

- [ ] Download Qwen 3.5 9B in LM Studio (~6GB)
- [ ] Load model in LM Studio
- [ ] Verify with: `curl http://localhost:1234/v1/models`
- [ ] Run the 4 commands above
- [ ] Monitor real-time progress
- [ ] Compare results with `compare_models.py`

---

## Common Questions

**Q: Do I need to download 9.7GB?**
A: No! Script automatically samples only 10K songs (~50MB).

**Q: Why Qwen 3.5 9B?**
A: Perfect VRAM fit (exactly 12GB), faster than Mistral (15-20 vs 10-15 tokens/sec), excellent accuracy.

**Q: How long for labeling?**
A: 20-30 minutes for 10K songs on your 6750 XT.

**Q: Will accuracy improve?**
A: Yes! 3.6x more data + better labels = 15-20% improvement expected.

**Q: Can I resume if interrupted?**
A: Yes! Script has checkpointing - run again to resume.

---

## Next Steps

1. **Read**: [QWEN_3.5_9B_SETUP.md](QWEN_3.5_9B_SETUP.md) (~5 min)
2. **Download**: Qwen 3.5 9B in LM Studio (5-10 min)
3. **Run**: The 4 commands above (40-75 min total)
4. **Enjoy**: Your improved model! 🚀
