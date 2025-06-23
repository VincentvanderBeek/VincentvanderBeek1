import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
from sys import stdout
import torch
import torchaudio
import pandas as pd
import csv
from collections import Counter
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSequenceClassification, pipeline
)

# Config
EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv")
DEFAULT_TRACK_INDEX = 0
CHUNK_DURATION = 30
OVERLAP_DURATION = 3
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
EMOTION_MODEL = "borisn70/bert-43-multilabel-emotion-detection"
BERT_TO_CORE = {
    'admiration': 'happiness', 'amusement': 'happiness', 'anger': 'anger', 'annoyance': 'anger',
    'approval': 'happiness', 'caring': 'happiness', 'confusion': 'fear', 'curiosity': 'neutral',
    'desire': 'happiness', 'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'disgust',
    'embarrassment': 'sadness', 'excitement': 'happiness', 'fear': 'fear', 'gratitude': 'happiness',
    'grief': 'sadness', 'joy': 'happiness', 'love': 'happiness', 'nervousness': 'fear',
    'optimism': 'happiness', 'pride': 'happiness', 'realization': 'neutral', 'relief': 'happiness',
    'remorse': 'sadness', 'sadness': 'sadness', 'surprise': 'surprised', 'neutral': 'neutral',
    'worry': 'fear', 'happiness': 'happiness', 'fun': 'happiness', 'hate': 'anger', 'autonomy': 'neutral',
    'safety': 'neutral', 'understanding': 'neutral', 'empty': 'sadness', 'enthusiasm': 'happiness',
    'recreation': 'happiness', 'sense of belonging': 'happiness', 'meaning': 'neutral',
    'sustenance': 'neutral', 'creativity': 'neutral', 'boredom': 'sadness'
}

# Utility functions for timing and progress
def log_step(step_name, start_time=None):
    """Log the start or completion of a step with timing"""
    current_time = time.time()
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if start_time is None:
        print(f"[{timestamp}] Starting: {step_name}")
        return current_time
    else:
        duration = current_time - start_time
        print(f"[{timestamp}] Completed: {step_name} (took {duration:.2f}s)")
        return current_time

def format_duration(seconds):
    """Format duration in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

def print_progress_bar(current, total, width=50):
    """Print a progress bar"""
    if total == 0:
        return
    
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    print(f"\r[{bar}] {percentage:.1f}% ({current}/{total})", end='', flush=True)

# Functions

def get_video_duration(video_path):
    step_start = log_step("Getting video duration")
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())
    log_step("Getting video duration", step_start)
    print(f"Video duration: {format_duration(duration)}")
    return duration

def extract_audio(video_path, track_index, output_path):
    step_start = log_step(f"Extracting audio (track {track_index})")
    
    video_duration = get_video_duration(video_path)
    tmp_path = str(output_path) + "_tmp.wav"
    
    print(f"Extracting audio from: {Path(video_path).name}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats",
           "-i", video_path, "-y", "-vn", "-c:a", "pcm_s16le",
           "-map", f"0:a:{track_index}", tmp_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Error extracting track {track_index}")
        return None

    # Check if padding is needed
    pad_start = log_step("Checking audio duration and padding if needed")
    audio_duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", tmp_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip())

    if audio_duration < video_duration - 1:
        print(f"Padding audio: {format_duration(audio_duration)} → {format_duration(video_duration)}")
        pad_cmd = ["ffmpeg", "-y", "-i", tmp_path,
                   "-filter_complex", f"[0]apad=pad_dur={video_duration - audio_duration}",
                   "-c:a", "pcm_s16le", str(output_path)]
        subprocess.run(pad_cmd, check=True)
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, output_path)
    
    log_step("Checking audio duration and padding if needed", pad_start)
    log_step(f"Extracting audio (track {track_index})", step_start)
    print(f"Audio extracted to: {Path(output_path).name}")
    return datetime.now()

def calculate_correct_timestamp(global_time, base_time):
    corrected = base_time + timedelta(seconds=global_time)
    return corrected.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], int(corrected.timestamp())

def transcribe_chunks(audio_path, model_name, video_start_time):
    step_start = log_step("Transcribing audio chunks")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    sample_rate = 16000

    stride = CHUNK_DURATION - OVERLAP_DURATION
    total_samples = waveform.size(1)
    total_duration = total_samples / sample_rate
    results, start_time, seen = [], 0.0, set()

    total_chunks = int((total_duration + stride - 1) // stride)
    chunk_idx = 0

    while start_time < total_duration:
        chunk_idx += 1
        end_time = min(start_time + CHUNK_DURATION, total_duration)
        start_sample = int(max(0, (start_time - OVERLAP_DURATION) * sample_rate))
        end_sample = int(end_time * sample_rate)
        chunk = waveform[:, start_sample:end_sample]

        # Replace print lines with progress bar
        progress = chunk_idx / total_chunks
        bar_width = 40
        filled_len = int(bar_width * progress)
        bar = '█' * filled_len + '-' * (bar_width - filled_len)
        stdout.write(f'\rTranscribing chunks: |{bar}| {chunk_idx}/{total_chunks} ({progress*100:.1f}%)')
        stdout.flush()

        inputs = processor(
            chunk.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")

        with torch.no_grad():
            ids = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                forced_decoder_ids=forced_decoder_ids
            )

        text = processor.decode(ids[0], skip_special_tokens=True)

        for sentence in text.split('.'):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                timestamp, unix_time = calculate_correct_timestamp(start_time, video_start_time)
                results.append([timestamp, unix_time, sentence])
                seen.add(sentence)
                start_time += len(sentence.split()) / 3.0  # Estimate duration by 3 wps heuristic

        start_time += stride

    # Newline after progress bar done
    print()

    end_time, end_unix = calculate_correct_timestamp(total_duration, video_start_time)
    results.append([end_time, end_unix, "end transcribe"])

    log_step("Transcribing audio chunks", step_start)
    return results

def save_csv(rows, path):
    step_start = log_step("Saving transcription CSV")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "unix_time", "transcript"])
        writer.writerows(rows)
    log_step("Saving transcription CSV", step_start)
    print(f"Transcription saved to: {Path(path).name}")

def load_emotion_model():
    step_start = log_step("Loading emotion detection model")
    tok = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    mod = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    log_step("Loading emotion detection model", step_start)
    return tok, mod

def classify_emotions(df, text_col, tok, mod):
    step_start = log_step("Classifying emotions")
    texts = df[text_col].dropna().astype(str).tolist()
    
    print(f"Analyzing emotions for {len(texts)} text segments")
    pipe = pipeline("text-classification", model=mod, tokenizer=tok, top_k=None, truncation=True)
    
    # Process in batches to show progress
    batch_size = 10
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = pipe(batch)
        all_results.extend(batch_results)
        print_progress_bar(min(i + batch_size, len(texts)), len(texts))
    
    print()  # New line after progress bar
    
    output = []
    for res in all_results:
        top = max(res, key=lambda x: x["score"])
        core = BERT_TO_CORE.get(top["label"], "neutral")
        output.append((top["label"], core, top["score"]))

    df = df.loc[df[text_col].notnull()].reset_index(drop=True).copy()
    df[["BERT_emotion", "core_emotion", "confidence"]] = output
    
    log_step("Classifying emotions", step_start)
    return df

def merge_emotions(transcript_df, merged_csv_path):
    # Define a safe_mode function for dominant emotion calculation
    def safe_mode(series):
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return None
        if len(counts) == 1:
            return counts.index[0]
        if 'neutral' in counts.index and len(counts) > 1:
            counts = counts.drop('neutral')
        return counts.idxmax()

    # Create aligned timestamps from transcript_df and merged_csv_path
    merged_df = pd.read_csv(merged_csv_path)
    transcript = transcript_df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(transcript['timestamp']):
        transcript['timestamp'] = pd.to_datetime(transcript['timestamp'])
    if 'datetime' in merged_df.columns and not pd.api.types.is_datetime64_any_dtype(merged_df['datetime']):
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

    # Create unix timestamps
    transcript['unix_timestamp'] = transcript['timestamp'].astype('int64') // 10**9
    merged_df['unix_timestamp'] = merged_df['datetime'].astype('int64') // 10**9

    # Calculate dominant emotion per second
    dominant_emotions = merged_df.groupby('unix_timestamp')['emotion'].apply(safe_mode)
    merged_df['dominant_emotion'] = merged_df['unix_timestamp'].map(dominant_emotions)

    # Align timestamps based on the latest point in time
    latest_merge_unix = merged_df['unix_timestamp'].max()
    latest_transcript_unix = transcript['unix_timestamp'].max()
    diff = latest_merge_unix - latest_transcript_unix

    # Shift transcript to align
    transcript['aligned_unix_timestamp'] = transcript['unix_timestamp'] + diff
    transcript['unix_time_rounded'] = transcript['aligned_unix_timestamp'].round().astype(int)

    # Build timestamp → emotion mapping
    timestamp_to_emotion = merged_df.set_index('unix_timestamp')['dominant_emotion'].to_dict()

    # Apply mapping with fallback to "Neutral"
    transcript['mapped_emotion'] = transcript['unix_time_rounded'].apply(
        lambda ts: timestamp_to_emotion.get(ts, "Neutral")
    )

    # Ensure all required columns are present
    required_cols = ["timestamp", "unix_time_rounded", "transcript", "BERT_emotion", "core_emotion", "mapped_emotion"]
    for col in required_cols:
        if col not in transcript.columns:
            transcript[col] = pd.NA

    return transcript[required_cols]

def create_enhanced_merge(transcript_df, merged_csv_path, output_path):
    """Create an enhanced merged CSV file with aligned transcript and emotion windows."""
    step_start = log_step("Creating enhanced merged file")
    
    # Load and copy merged_df
    merged_df = pd.read_csv(merged_csv_path)
    enhanced_df = merged_df.copy()

    print(f"Enhancing: {Path(merged_csv_path).name}")
    print(f"Original entries: {len(enhanced_df)}")

    # Ensure correct types
    if 'datetime' in enhanced_df.columns and not pd.api.types.is_datetime64_any_dtype(enhanced_df['datetime']):
        enhanced_df['datetime'] = pd.to_datetime(enhanced_df['datetime'])
    if not pd.api.types.is_datetime64_any_dtype(transcript_df['timestamp']):
        transcript_df['timestamp'] = pd.to_datetime(transcript_df['timestamp'])

    # Create aligned unix timestamps for both
    transcript_df['unix_timestamp'] = transcript_df['timestamp'].astype('int64') // 10**9
    enhanced_df['unix_timestamp'] = enhanced_df['datetime'].astype('int64') // 10**9

    # Align timelines
    latest_transcript = transcript_df['unix_timestamp'].max()
    latest_merged = enhanced_df['unix_timestamp'].max()
    diff = latest_merged - latest_transcript

    # Shift transcript to match merged file
    transcript_df['aligned_unix_timestamp'] = transcript_df['unix_timestamp'] + diff
    transcript_df['window'] = (transcript_df['aligned_unix_timestamp'] // 3) * 3
    
    # Build 3-second window summaries
    def get_window_transcript(transcripts):
        filtered = [t for t in transcripts if pd.notna(t) and str(t).strip().lower() != "end transcribe"]
        return " ".join(filtered).strip()

    def get_window_core_emotion(emotions):
        filtered = [e for e in emotions if pd.notna(e)]
        return Counter(filtered).most_common(1)[0][0].lower() if filtered else "neutral"

    window_to_transcript = transcript_df.groupby("window")["transcript"].apply(get_window_transcript).to_dict()
    window_to_core_emotion = transcript_df.groupby("window")["core_emotion"].apply(get_window_core_emotion).to_dict()

        # Map these to the merged_df using same window logic
    def lookup_window_data(ts, lookup):
        window = (ts // 3) * 3
        return lookup.get(window, "" if lookup is window_to_transcript else "neutral")

    enhanced_df['transcript'] = enhanced_df['unix_timestamp'].apply(lambda ts: lookup_window_data(ts, window_to_transcript))
    enhanced_df['core_emotion'] = enhanced_df['unix_timestamp'].apply(lambda ts: lookup_window_data(ts, window_to_core_emotion))

    # Write out result
    enhanced_df.to_csv(output_path, index=False)
    print(f"Enhanced file saved to {output_path}")
    
    log_step("Enhanced file created", step_start)
    
    return enhanced_df

# Main
def main(video_file):
    total_start = time.time()
    print("="*60)
    print(f"STARTING PIPELINE: {Path(video_file).name}")
    print("="*60)
    
    video_path = Path(video_file).resolve()
    
    # Create output folder
    output_folder = video_path.parent / f"{video_path.stem}_output"
    output_folder.mkdir(exist_ok=True)
    print(f"Output folder: {output_folder.name}")
    
    # Update all file paths to be inside the output folder
    audio_path = output_folder / "track0.wav"
    transcription_path = output_folder / "transcription.csv"
    final_output_path = output_folder / "merged_emotions.csv"
    enhanced_merge_path = output_folder / "enhanced_merged.csv"

    # Step 1: Audio extraction
    print(f"\nSTEP 1: AUDIO EXTRACTION")
    print("-" * 40)
    video_start = extract_audio(str(video_path), DEFAULT_TRACK_INDEX, str(audio_path))
    if not video_start:
        print("Audio extraction failed!")
        return

    # Step 2: Transcription
    print(f"\nSTEP 2: SPEECH TRANSCRIPTION")
    print("-" * 40)
    transcription = transcribe_chunks(str(audio_path), WHISPER_MODEL, video_start)
    save_csv(transcription, transcription_path)

    # Step 3: Emotion classification
    print(f"\nSTEP 3: EMOTION CLASSIFICATION")
    print("-" * 40)
    df = pd.read_csv(transcription_path)
    tok, mod = load_emotion_model()
    emotion_df = classify_emotions(df, "transcript", tok, mod)

    # Step 4: Merge with existing data
    print(f"\nSTEP 4: MERGING DATA")
    print("-" * 40)
    merged_files = list(video_path.parent.glob(f"{video_path.stem[:25]}*merged.csv"))
    if not merged_files:
        print("No merged.csv found.")
        return

    merged_result = merge_emotions(emotion_df, merged_files[0])
    
    # Step 5: Save final result
    save_start = log_step("Saving final merged file")
    merged_result.to_csv(final_output_path, index=False)
    log_step("Saving final merged file", save_start)
    
    # Step 6: Create enhanced merged file
    print(f"\nSTEP 5: CREATING ENHANCED MERGED FILE")
    print("-" * 40)
    enhanced_result = create_enhanced_merge(emotion_df, merged_files[0], enhanced_merge_path)
    
    # Summary
    total_duration = time.time() - total_start
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Output folder: {output_folder}")
    print(f"Transcript + emotions: {final_output_path.name}")
    print(f"Enhanced merged file: {enhanced_merge_path.name}")
    print(f"Total entries in transcript file: {len(merged_result)}")
    print(f"Total entries in enhanced file: {len(enhanced_result)}")
    print(f"Total time: {format_duration(total_duration)}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python STT_EMOTION_pipeline.py <video_file>")
    else:
        main(sys.argv[1])