"""
Download and process Kaggle Genius Song Lyrics dataset efficiently using the kaggle library.
Samples 10K songs to avoid downloading full 9.7GB.

SETUP REQUIRED:
1. Install: pip install kaggle
2. Setup credentials: https://www.kaggle.com/account
   - Download API token from account settings
   - Save to ~/.kaggle/kaggle.json
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE importing kaggle to ensure credentials are available
load_dotenv()

import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"
DATA_DIR = Path("data_generation/kaggle_raw")
PROCESSED_FILE = Path("data_generation/kaggle_lyrics_sample.csv")

def download_kaggle_dataset_chunked(sample_size=10000):
    """
    Download Kaggle dataset using the kaggle Python library and sample N songs.
    
    Requires kaggle API credentials at ~/.kaggle/kaggle.json
    Get token from: https://www.kaggle.com/account
    """
    print(f"Connecting to Kaggle API...")

    # If credentials are present as env vars, write them to ~/.kaggle/kaggle.json
    env_user = os.getenv("KAGGLE_USERNAME")
    env_key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_KEY")
    if env_user and env_key:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        try:
            with open(kaggle_json, "w", encoding="utf-8") as f:
                json.dump({"username": env_user, "key": env_key}, f)
            try:
                kaggle_json.chmod(0o600)
            except Exception:
                # chmod may fail on some filesystems; ignore
                pass
            print(f"✓ Wrote Kaggle credentials to {kaggle_json}")
        except Exception as e:
            print(f"✗ Failed to write kaggle.json: {e}")

    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        print("✓ Authenticated with Kaggle API\n")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("Setup required:")
        print("  1. Install: pip install kaggle")
        print("  2. Get token: https://www.kaggle.com/account")
        print("  3. Save to: ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY in .env")
        return False
    
    print(f"Downloading {KAGGLE_DATASET}...")
    print(f"(This may take 5-15 minutes for ~9.7GB)\n")
    
    try:
        # Download dataset
        api.dataset_download_files(
            KAGGLE_DATASET, 
            path=str(DATA_DIR), 
            unzip=True,
            quiet=False
        )
        print("\n✓ Download and extraction complete!")
        
        # Clean up zip file if it exists
        zip_files = list(DATA_DIR.glob("*.zip"))
        for zip_file in zip_files:
            zip_file.unlink()
            print(f"✓ Cleaned up {zip_file.name}")
            
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def process_kaggle_data_sampled(sample_size=10000):
    """Process Kaggle CSV with sampling to reduce data size."""
    print(f"Processing Kaggle data (sampling {sample_size} songs)...")
    
    # Find the main lyrics CSV
    raw_csv = None
    for csv_file in DATA_DIR.glob("*.csv"):
        if "lyric" in csv_file.name.lower():
            raw_csv = csv_file
            break
    
    if not raw_csv:
        print(f"No lyrics CSV found in {DATA_DIR}")
        return None
    
    print(f"Reading {raw_csv}...")
    # Read CSV with sampling for memory efficiency
    try:
        # Try multiple encodings if utf-8 fails
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(raw_csv, encoding=encoding, on_bad_lines='skip')
                print(f"  Successfully read with {encoding} encoding")
                break
            except Exception as enc_err:
                continue
        
        if df is None:
            print(f"Could not read CSV with any encoding")
            return None
            
        print(f"  Total songs in dataset: {len(df)}")
        
        # Sample N songs randomly
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"  Sampled: {len(df)} songs")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # Normalize column names
    column_mapping = {
        'song': 'song',
        'title': 'song',
        'name': 'song',
        'lyric': 'lyrics',
        'lyrics_text': 'lyrics',
        'artist_name': 'artist',
        'artist': 'artist',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Keep only song and lyrics
    keep_cols = [col for col in ['song', 'lyrics'] if col in df.columns]
    if len(keep_cols) < 2:
        print(f"Error: Missing required columns. Found: {df.columns.tolist()}")
        return None
    
    df = df[keep_cols]
    
    # Clean data
    df['song'] = df['song'].astype(str).str.strip()
    df['lyrics'] = df['lyrics'].astype(str).str.strip()
    
    # Remove empty lyrics
    df = df[df['lyrics'].str.len() > 0]
    df = df.drop_duplicates(subset=['song'], keep='first')
    
    print(f"✓ Cleaned dataset: {len(df)} songs")
    
    # Save processed data
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✓ Saved to {PROCESSED_FILE}")
    
    
    return df

def main():
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Kaggle Dataset Download - Using Kaggle Library")
    print("="*70)
    print(f"Target: 10,000 songs (avoiding full 9.7GB download)\n")
    
    # Check if zip needs to be extracted
    zip_files = list(DATA_DIR.glob("*.zip"))
    if zip_files:
        print("Step 0: Extract Kaggle dataset")
        print("-" * 70)
        for zip_file in zip_files:
            print(f"Extracting {zip_file.name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print(f"✓ Extraction complete")
                # Don't delete the zip yet, in case we need it
            except Exception as e:
                print(f"✗ Extraction failed: {e}")
                return False
        print()
    
    # Check if already downloaded
    existing_csv = None
    for csv_file in DATA_DIR.glob("*.csv"):
        if "lyric" in csv_file.name.lower():
            existing_csv = csv_file
            break
    
    if not existing_csv:
        print("Step 1: Download dataset from Kaggle")
        print("-" * 70)
        success = download_kaggle_dataset_chunked(sample_size=10000)
        
        if not success:
            print("\n✗ Download failed. Exiting.")
            return False
            
        print("\nStep 2: Verify download")
        print("-" * 70)
        existing_csv = None
        for csv_file in DATA_DIR.glob("*.csv"):
            if "lyric" in csv_file.name.lower():
                existing_csv = csv_file
                break
        
        if not existing_csv:
            print("✗ No lyrics CSV found after download")
            return False
    else:
        print(f"✓ Found existing CSV: {existing_csv}\n")
    
    # Process with sampling
    print("Step 3: Sample and process data")
    print("-" * 70)
    df = process_kaggle_data_sampled(sample_size=10000)
    
    if df is not None:
        print(f"\n{'='*70}")
        print(f"✓ Dataset ready for labeling!")
        print(f"  Songs: {len(df):,}")
        print(f"  File size: ~{PROCESSED_FILE.stat().st_size / (1024*1024):.1f}MB")
        print(f"  Location: {PROCESSED_FILE}")
        print(f"{'='*70}")
        return True
    else:
        print("✗ Processing failed")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
