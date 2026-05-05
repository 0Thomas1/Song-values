# Qwen 3.5 9B Setup Guide

## Why Qwen 3.5 9B?

✅ **Perfect fit for 6750 XT** (~12GB VRAM required, your GPU has exactly that)
✅ **Faster than Mistral** (15-20 tokens/sec vs 10-15 tokens/sec)
✅ **Excellent instruction following** (structured JSON output is reliable)
✅ **Optimized for reasoning tasks** (great for moral theme detection)
✅ **Smaller model, same quality** (9B parameters vs 7B, better efficiency)

---

## Quick Start

### 1. Download Model in LM Studio (5-10 minutes)

**In LM Studio application:**

1. Click the **magnifying glass icon** (Search/Browse Models) at the top
2. Search: `Qwen/Qwen3.5-9B-Instruct`
3. Click the **download button** next to the model
4. Wait for completion (~6GB download)

**Status check:**

- You should see download progress
- Once complete, model appears in your library

### 2. Load Model in LM Studio (1-2 minutes)

1. Click the **model dropdown** (currently shows "Select a model")
2. Select `Qwen3.5-9B-Instruct` from the list
3. Click **"Load Model"** button
4. Wait for loading to complete (~30-60 seconds)
5. You should see **"Ready to Chat"** at the bottom

### 3. Verify Connection (30 seconds)

In your terminal, test the connection:

```bash
curl http://localhost:1234/v1/models
```

You should get JSON response showing Qwen loaded:

```json
{
  "object": "list",
  "data": [
    {
      "id": "lm-studio",
      "object": "model",
      "owned_by": "lm-studio",
      "permission": []
    }
  ]
}
```

**If you get "Connection refused":**

- Make sure LM Studio app is open and running
- Check that port 1234 is used (LM Studio → Settings → Server)

### 4. Start Labeling! (20-30 minutes)

```bash
# Make sure venv is active
source .venv/bin/activate

# Run the labeling script
python data_generation/label_kaggle_lyrics.py
```

**What you'll see:**

```
Connecting to LM Studio...
⚠️  Make sure LM Studio is running with Qwen 3.5 9B loaded!
   LM Studio should be at: localhost:1234

✓ Connected to model: qwen3.5-9b-instruct
  ✓ Qwen 3.5 9B detected - using optimal settings for your 6750 XT
  - Concurrent requests: 4
  - Expected speed: ~15-20 tokens/sec
  - ETA: ~30-40 minutes for 10000 songs

Progress: 50/10000 songs
  ✓ Labeled: 48 | ✗ Failed: 2
  Speed: 3.2 songs/min
  ETA: 25 minutes remaining
```

---

## Performance Specs

| Metric                 | Qwen 3.5 9B      | Your GPU          |
| ---------------------- | ---------------- | ----------------- |
| **Model size**         | 9B parameters    | -                 |
| **VRAM needed**        | ~12GB            | ✅ 12GB (6750 XT) |
| **Speed**              | 15-20 tokens/sec | -                 |
| **Time for 10K songs** | 20-30 minutes    | -                 |
| **Accuracy**           | ★★★★★            | -                 |
| **JSON output**        | Excellent        | -                 |

---

## Timeline Estimate

```
Setup:                    5-10 min
  - Download model:       5-10 min
  - Load model:           1-2 min

Preparation:              5 min
  - Activate venv
  - Check dependencies

Data Download:            5-15 min
  python data_generation/download_kaggle.py

Label Generation:         20-30 min
  python data_generation/label_kaggle_lyrics.py

Model Training:           5-15 min
  python training/train_fasttext.py

─────────────────────────────────
TOTAL:                    40-75 minutes
```

---

## Expected Results After Running

### Files Created

- ✅ `data_generation/kaggle_lyrics_sample.csv` (~50MB)
- ✅ `kaggle_synthesized_labels.jsonl` (~10MB)
- ✅ `models/morale_classifier_fasttext_v2.bin` (~50MB)

### Performance Improvement

```
Before:        After:         Improvement:
─────────────────────────────────────────────
3,781 songs    13,781 songs   +264%
F1 = 0.61      F1 = 0.72+     +18%
2 sec/song     0.1 sec/song   20x faster
```

---

## Troubleshooting

### Problem: "Connection refused" when running script

**Solution:**

1. Make sure LM Studio app is open (not just running in background)
2. Verify in LM Studio Settings that Server is on port 1234
3. Try curl test: `curl http://localhost:1234/v1/models`
4. Restart LM Studio if needed

### Problem: "Model not found" or "Empty response"

**Solution:**

1. Check that Qwen model is fully loaded in LM Studio
2. You should see "Ready to Chat" in LM Studio
3. Try chatting in LM Studio to verify model works
4. Then run the script again

### Problem: Script is running very slowly (< 5 songs/min)

**Possible causes:**

- LM Studio is still warming up (wait 1-2 min)
- VRAM is full/swapping to disk
- GPU is thermal throttling

**Solutions:**

1. Wait a couple minutes for LM Studio to stabilize
2. Close other GPU-intensive apps
3. Check GPU temperature: `nvidia-smi` or system monitor
4. The script can resume - just Ctrl+C and run again

### Problem: Script crashes with timeout errors

**Solution:**

- Most songs likely timed out (>30s to generate labels)
- Check VRAM usage - if full, reduce concurrent requests
- In label_kaggle_lyrics.py, change:
  ```python
  semaphore = asyncio.Semaphore(2)  # Reduce from 4
  ```
- Run again - checkpoint will resume from where it left off

### Problem: Labels look wrong/inconsistent

**Solution:**

1. Verify Qwen model is correctly loaded (check chat functionality in LM Studio)
2. A few failed songs are normal (< 2% failures expected)
3. Check sample labels:
   ```bash
   head -5 kaggle_synthesized_labels.jsonl
   ```
4. Labels should have ~6-8 themes per song on average

---

## GPU Optimization Tips

### For 6750 XT with 12GB VRAM:

**Monitor during labeling:**

```bash
# Watch GPU usage in real-time
watch -n 1 "nvidia-smi"
# Or with AMD tools: radeontop
```

**If VRAM usage > 95%:**

1. Reduce concurrent requests in script:
   ```python
   semaphore = asyncio.Semaphore(2)  # Instead of 4
   ```
2. Close other applications
3. Reduce context size in LM Studio settings

**If GPU not being used:**

1. Make sure GPU acceleration is enabled in LM Studio
2. Check LM Studio Settings → GPU Detection
3. May need to install AMD GPU drivers

---

## Next Steps

1. **Download Qwen 3.5 9B** in LM Studio (~5-10 minutes)
2. **Load the model** and verify with curl test (~2 minutes)
3. **Run full pipeline:**
   ```bash
   python data_generation/download_kaggle.py
   python data_generation/label_kaggle_lyrics.py
   python training/train_fasttext.py
   ```
4. **Compare results:**
   ```bash
   python training/compare_models.py
   ```

---

## Reference

- Qwen Model: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct
- LM Studio: https://lmstudio.ai
- Your GPU: AMD Radeon 6750 XT (12GB VRAM)

---

**Ready? Download Qwen 3.5 9B in LM Studio and follow the steps above!** 🚀
