"""
Generate moral labels for Kaggle lyrics using LM Studio.

MODEL: Qwen 3.5 9B
- Speed: ~15-20 tokens/sec on AMD 6750 XT = ~20-30 min for 10K songs
- Accuracy: Excellent (★★★★★)
- Memory: ~12GB (perfect fit for 12GB VRAM)
- Download: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct

WHY QWEN 3.5 9B:
- Best instruction following for structured JSON output
- Optimal VRAM usage for your 6750 XT
- Fast enough for practical use (~15-20 tokens/sec)
- Excellent accuracy for complex reasoning tasks like moral theme detection

HOW TO USE:
1. Download model in LM Studio: search 'Qwen/Qwen3.5-9B-Instruct'
2. Load it in LM Studio (will download automatically ~6GB)
3. Make sure LM Studio is running on localhost:1234
4. Run this script
"""

import pandas as pd
import asyncio
import json
import os
from pathlib import Path
import dotenv
import aiofiles
from pydantic import BaseModel
import lmstudio as lms

dotenv.load_dotenv()

class MoralLabels(BaseModel):
    Love: bool
    Hate: bool
    Joy: bool
    Despair: bool
    Peace: bool
    Conflict: bool
    Patience: bool
    Impatience: bool
    Kindness: bool
    Cruelty: bool
    Goodness: bool
    Malice: bool
    Faithfulness: bool
    Betrayal: bool
    Gentleness: bool
    Harshness: bool
    Self_control: bool
    Recklessness: bool

# System instruction to guide the model
SYSTEM_INSTRUCTION = """
You are an expert lyrical and thematic analyzer specializing in identifying moral and emotional themes in song lyrics.

Analyze the provided lyrics for the presence of exactly 18 specific moral and emotional themes. For each theme, determine if it is:
- EXPLICITLY present (directly stated or clearly implied)
- STRONGLY IMPLICITLY present (evident through metaphor, symbolism, or narrative)
- ABSENT (not meaningfully present)

Guidelines for accuracy:
1. Be thorough but fair - look for genuine thematic evidence
2. Opposing themes (e.g., Love vs Hate) can BOTH be true if the song explores internal conflict or contrast
3. Consider metaphor, symbolism, narrative arc, and emotional tone
4. A theme is "true" if a listener would reasonably identify it as part of the song's message
5. Don't force a theme if it's not genuinely present - be conservative

Format: Return ONLY valid JSON matching the provided schema.
"""

PROCESSED_CSV = "data_generation/kaggle_lyrics.csv"
OUTPUT_JSONL = "kaggle_synthesized_labels.jsonl"
CHECKPOINT_FILE = "data_generation/.kaggle_checkpoint.json"

async def analyze_song(song_id: str, lyrics: str, semaphore: asyncio.Semaphore, output_file: str, model):
    """Analyzes a single song with rate-limiting and auto-saving."""
    async with semaphore:
        try:
            # Truncate lyrics to avoid context overflow (Mistral can handle 4K tokens)
            MAX_LYRIC_CHARS = 3000
            cleaned_lyrics = " ".join(lyrics.split())
            if len(cleaned_lyrics) > MAX_LYRIC_CHARS:
                cleaned_lyrics = cleaned_lyrics[:MAX_LYRIC_CHARS]
            
            prompt = f"{SYSTEM_INSTRUCTION}\n\nLyrics:\n{cleaned_lyrics}"
            
            # Call LM Studio with structured output
            # Timeout after 30 seconds per song
            response = await asyncio.wait_for(
                model.respond(prompt, response_format=MoralLabels),
                timeout=30.0
            )
            
            labels = response.parsed
            if not labels:
                raise ValueError(f"Empty response for song {song_id}")
            
            result = {"song_id": song_id, **labels}
            
            # CHECKPOINTING: Append to JSONL file immediately
            async with aiofiles.open(output_file, 'a') as f:
                await f.write(json.dumps(result) + '\n')
            
            return result
            
        except asyncio.TimeoutError:
            print(f"⚠ Timeout on song ID {song_id} (took >30s)")
            return None
        except Exception as e:
            print(f"⚠ Failed on song ID {song_id}. Error: {str(e)[:100]}")
            return None

def load_checkpoint():
    """Load checkpoint to resume processing."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed": 0, "failed": 0}

def save_checkpoint(processed, failed):
    """Save checkpoint for resuming."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"processed": processed, "failed": failed}, f)

async def main():
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_count = checkpoint["processed"]
    failed_count = checkpoint["failed"]
    
    # Load CSV data
    print(f"Loading {PROCESSED_CSV}...")
    df = pd.read_csv(PROCESSED_CSV)
    
    # Skip already processed rows
    start_idx = processed_count + failed_count
    if start_idx > 0:
        print(f"Resuming from row {start_idx}...")
        df = df.iloc[start_idx:].reset_index(drop=True)
    
    print(f"Processing {len(df)} songs...")
    
    # Initialize LM Studio client
    print("Connecting to LM Studio...")
    print("⚠️  Make sure LM Studio is running with Qwen 3.5 9B loaded!")
    print("   LM Studio should be at: localhost:1234\n")
    
    async with lms.AsyncClient() as client:
        model = await client.llm.model()
        model_name = getattr(model, 'name', 'Unknown Model')
        print(f"✓ Connected to model: {model_name}")
        
        # Recommend optimal settings for Qwen
        if "qwen" in model_name.lower():
            print("  ✓ Qwen 3.5 9B detected - using optimal settings for your 6750 XT")
            semaphore = asyncio.Semaphore(4)  # 4 concurrent (Qwen handles well)
            print("  - Concurrent requests: 4")
            print("  - Expected speed: ~15-20 tokens/sec")
            print(f"  - ETA: ~{len(df) // 300:.0f}-{len(df) // 200:.0f} minutes for {len(df)} songs\n")
        else:
            print(f"  Model: {model_name}")
            print("  ⚠️  Not Qwen 3.5 9B - adjust settings if needed")
            semaphore = asyncio.Semaphore(3)
        
        # Create tasks for all songs
        tasks = [
            analyze_song(
                song_id=row['song'],
                lyrics=row['lyrics'],
                semaphore=semaphore,
                output_file=OUTPUT_JSONL,
                model=model
            )
            for _, row in df.iterrows()
        ]
        
        # Process with progress updates
        completed = 0
        start_time = None
        for task in asyncio.as_completed(tasks):
            if start_time is None:
                import time
                start_time = time.time()
            
            result = await task
            completed += 1
            if result:
                processed_count += 1
            else:
                failed_count += 1
            
            if completed % 50 == 0:
                import time
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (len(tasks) - completed) / rate if rate > 0 else 0
                
                print(f"Progress: {completed}/{len(tasks)} songs")
                print(f"  ✓ Labeled: {processed_count} | ✗ Failed: {failed_count}")
                print(f"  Speed: {rate:.1f} songs/min")
                print(f"  ETA: {remaining/60:.0f} minutes remaining\n")
                
                save_checkpoint(processed_count, failed_count)
        
        save_checkpoint(processed_count, failed_count)
    
    print(f"\n✓ Labeling complete!")
    print(f"  Total processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output: {OUTPUT_JSONL}")

if __name__ == "__main__":
    asyncio.run(main())
