"""
THIS SCRIPT IS USELESS NOW
UNLESS YOU WANT TO SYNTHESISE LABELS FOR A NEW DATASET
"""
import pandas as pd
import asyncio
import json
import os
import dotenv
import aiofiles
from pydantic import BaseModel
import lmstudio as lms

# 1. Define the Strict JSON Schema using Pydantic
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

# System Instruction to guide the model
SYSTEM_INSTRUCTION = """
You are an expert lyrical analyzer. Evaluate the presence of 18 specific moral and emotional themes in the provided lyrics.
For each theme, respond 'true' if it is explicitly or strongly implicitly present, 'false' if absent.
Return the result as JSON matching the schema provided.
"""

async def analyze_song(song_id: str, lyrics: str, semaphore: asyncio.Semaphore, output_file: str, model):
    """Analyzes a single song with rate-limiting and auto-saving."""
    # The semaphore ensures we don't overload LM Studio with too many concurrent requests
    async with semaphore:
        try:
            # Truncate lyrics to avoid context overflow
            MAX_LYRIC_CHARS = 2000
            cleaned_lyrics = " ".join(lyrics.split())
            if len(cleaned_lyrics) > MAX_LYRIC_CHARS:
                cleaned_lyrics = cleaned_lyrics[:MAX_LYRIC_CHARS]
            
            prompt = f"{SYSTEM_INSTRUCTION}\n\nLyrics:\n{cleaned_lyrics}"
            
            # Call LM Studio Gemma model with structured output
            response = await model.respond(prompt, response_format=MoralLabels)
            
            # Extract parsed labels from LM Studio response
            labels = response.parsed
            if not labels:
                raise ValueError(f"Empty response for song {song_id}")
            
            # Combine the ID and the labels
            result = {"song_id": song_id, **labels}
            
            # CHECKPOINTING: Append to JSONL file immediately
            async with aiofiles.open(output_file, 'a') as f:
                await f.write(json.dumps(result) + '\n')
                
            print(f"Successfully processed song ID: {song_id}")
            return result
            
        except Exception as e:
            print(f"Failed on song ID {song_id}. Error: {e}")
            return None

async def main():
    # Initialize LM Studio client
    print("Connecting to LM Studio...")
    async with lms.AsyncClient() as client:
        model = await client.llm.model()
        print(f"Loaded model: {model}")
        
        # 2. Load your Kaggle Dataset
        print("Loading dataset...")
        df = pd.read_csv("data_generation/lyrics/all_lyrics.csv")
        
        # Optional: If you want to test it on 50 songs first before doing all 3880
        # df = df.head(50)

        output_file = "synthesized_labels.jsonl"
        
        # If the script stopped halfway previously, find out where we left off
        processed_ids = set()
        if os.path.exists(output_file):
            async with aiofiles.open(output_file, 'r') as f:
                contents = await f.read()
                for line in contents.strip().split('\n'):
                    if line:
                        processed_ids.add(json.loads(line)['song_id'])
            print(f"Found {len(processed_ids)} already processed songs. Resuming...")

        # 3. Create the tasks
        tasks = []
        # Limit concurrent LM Studio calls (adjust based on your GPU memory)
        semaphore = asyncio.Semaphore(10)
        
        for index, row in df.iterrows():
            song_id = str(row['song'])
            lyrics = str(row['lyrics'])
            
            # Skip if already processed
            if song_id in processed_ids:
                continue
                
            # Add to our task queue
            task = asyncio.create_task(analyze_song(song_id, lyrics, semaphore, output_file, model))
            tasks.append(task)
            
        print(f"Starting analysis of {len(tasks)} songs...")
        
        # 4. Run all tasks concurrently
        await asyncio.gather(*tasks)
        print("\nBatch processing complete!")

# Kick off the async loop
if __name__ == "__main__":
    asyncio.run(main())