import pandas as pd
import os
import tqdm
import torch
import torchaudio

def load_data(base_dir, batch_size, target_sr=16000):
    """
    Load, resample, and prepare audio data with corresponding labels.
    
    Inputs:
    base_dir (str): directory containing wav_files and labels.csv
    batch_size (int): n of samples to load
    target_sr (int): Target sample rate (by default 16kHz)
    
    Returns:
    data_list (list): Each element is a tuple (waveform, label_tensor)
    """

    # Load wav files dir and labels dir
    wav_dir = os.path.join(base_dir, "wav_files")
    labels_dir = os.path.join(base_dir, "labels.csv")

    # Load labels file
    labels_df = pd.read_csv(labels_dir)
        
    # Convert numeric filenames in CSV to string and append '.wav'
    labels_df['filename'] = labels_df['filename'].astype(str) + ".wav"

    # Get only actual .wav files in the directory (ignore other files in folder)
    wav_files = set(f for f in os.listdir(wav_dir) if f.lower().endswith(".wav"))

    # Keep only CSV rows where the .wav file actually exists
    labels_df = labels_df[labels_df['filename'].isin(wav_files)].reset_index(drop=True)

    # Choose size of batch
    labels_df = labels_df.head(batch_size)

    print(f" 1. Found {len(labels_df)} matching audio-label pairs.")

    # Load all waveforms
    data = []

    # Load waveforms and labels
    print("\n 2. Loading audio files")
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Loading audio files"):
        file_path = os.path.join(wav_dir, row['filename'])

        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)

        data.append({
            "filename": row['filename'],
            "waveform": waveform,
            "sample_rate": sample_rate,
            "valence": float(row['Klang Valence']),
            "arousal": float(row['Klang Arousal'])
        })
  
    print(f"\n 3. Resampling Waveforms to {target_sr}")
    for i, d in enumerate(data):
        if d["sample_rate"] != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=d["sample_rate"], new_freq=target_sr)
            d["waveform"] = resampler(d["waveform"])
            d["sample_rate"] = target_sr
        print(f"{i+1:03d}: {d['filename']} → {target_sr}Hz | Shape: {tuple(d['waveform'].shape)}")

    # Build torch dataset list
    data_list = []
    for d in data:
        label = torch.tensor([d["valence"], d["arousal"]], dtype=torch.float32)
        data_list.append((d["waveform"], label))

    print("\n Data ready as a Torch object")
    print(f"Total samples: {len(data_list)}\n")

    return data_list