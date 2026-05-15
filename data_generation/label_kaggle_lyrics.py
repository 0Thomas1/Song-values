"""
Generate moral labels for Kaggle lyrics using LM Studio.


HOW TO USE:
1. Download model in LM Studio
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

PROCESSED_CSV = "data_generation/kaggle_lyrics_sample.csv"
OUTPUT_JSONL = "kaggle_synthesized_labels.jsonl"
CHECKPOINT_FILE = "data_generation/.kaggle_checkpoint.json"

async def analyze_song(song_id: str, lyrics: str, semaphore: asyncio.Semaphore, output_file: str, model, timeout: float = 60.0):
    """Analyzes a single song with rate-limiting and auto-saving."""
    async with semaphore:
        try:
            # Truncate lyrics to avoid context overflow
            MAX_LYRIC_CHARS = 2000
            cleaned_lyrics = " ".join(lyrics.split())
            if len(cleaned_lyrics) > MAX_LYRIC_CHARS:
                cleaned_lyrics = cleaned_lyrics[:MAX_LYRIC_CHARS]
            
            prompt = f"{SYSTEM_INSTRUCTION}\n\nLyrics:\n{cleaned_lyrics}"
            
            # Call LM Studio with structured output
            # Increase timeout to 60 seconds for slower models
            response = await asyncio.wait_for(
                model.respond(prompt, response_format=MoralLabels),
                timeout=timeout
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
            print(f"⚠ Timeout on song ID {song_id} (took >{timeout}s)")
            return None
        except Exception as e:
            error_msg = str(e)[:100]
            # Skip spammy channel closed errors in output
            if "channel" not in error_msg.lower():
                print(f"⚠ Failed on song ID {song_id}. Error: {error_msg}")
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
    print("⚠️  Make sure LM Studio is running with a model loaded!")
    print("   LM Studio should be at: localhost:1234\n")
    
    async with lms.AsyncClient() as client:
        model = await client.llm.model()
        model_name = getattr(model, 'name', 'Unknown Model')
        print(f"✓ Connected to model: {model_name}")
        
        # Determine settings based on loaded model
        model_lower = model_name.lower()
        
        if "qwen" in model_lower:
            print("  ✓ Qwen detected - using Qwen-optimized settings")
            semaphore = asyncio.Semaphore(6)  # Qwen handles concurrency well
            timeout = 45.0  # Qwen is fast
            print(f"  - Concurrent requests: 6")
            print(f"  - Timeout: {timeout}s per song")
            print(f"  - Expected speed: ~15-20 tokens/sec")
            print(f"  - ETA: ~{len(df) // 250:.0f}-{len(df) // 150:.0f} minutes for {len(df)} songs\n")
        elif "mistral" in model_lower:
            print("  ✓ Mistral detected - using Mistral-optimized settings")
            semaphore = asyncio.Semaphore(8)  # Mistral can handle more concurrency
            timeout = 60.0  # Mistral is slower
            print(f"  - Concurrent requests: 8")
            print(f"  - Timeout: {timeout}s per song")
            print(f"  - Expected speed: ~8-12 tokens/sec")
            print(f"  - ETA: ~{len(df) // 150:.0f}-{len(df) // 80:.0f} minutes for {len(df)} songs\n")
        else:
            print(f"  ⚠️  Unknown model type: {model_name}")
            print("  Using conservative default settings")
            semaphore = asyncio.Semaphore(6)
            timeout = 90.0
            print(f"  - Concurrent requests: 8")
            print(f"  - Timeout: {timeout}s per song\n")
        
        # Create tasks for all songs
        tasks = [
            analyze_song(
                song_id=row['song'],
                lyrics=row['lyrics'],
                semaphore=semaphore,
                output_file=OUTPUT_JSONL,
                model=model,
                timeout=timeout
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
            
            if completed % 100 == 0:
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
