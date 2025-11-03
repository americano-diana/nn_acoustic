import pandas as pd
import os
import tqdm
import torch
import torchaudio
import numpy as np

def load_data(base_dir, batch_size, target_sr=16000, normalise_targets=False):
    """
    Load and prepare audio data with corresponding labels.
    
    Inputs:
    base_dir (str): directory containing wav_files and labels.csv
    batch_size (int): n of samples to load
    normalize_targets (bool): Whether to normalize valence/arousal targets (default: False)
    
    Returns:
    data_dict (dict): Contains waveforms, valences, arousals, filenames, and optionally normalization stats
    """

    # Load wav files dir and labels dir

    # Updated to load from pre-resampled folder
    wav_dir = os.path.join(base_dir, "wav_16k")
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
    for _, row in tqdm.tqdm(labels_df.iterrows(), total=len(labels_df), desc="Loading audio files"):
        file_path = os.path.join(wav_dir, row['filename'])

        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)

        # Quick check: ensure correct sample rate
        if sample_rate != target_sr:
            print(f" Warning: {row['filename']} has {sample_rate}Hz (expected {target_sr}Hz)")

        data.append({
            "filename": row['filename'],
            "waveform": waveform,
            "sample_rate": sample_rate,
            "valence": float(row['Klang Valence']),
            "arousal": float(row['Klang Arousal'])
        })

        # Extract targets before normalization
    valences = np.array([float(d["valence"]) for d in data])
    arousals = np.array([float(d["arousal"]) for d in data])

    # Optionally normalize targets
    target_stats = None
    if normalise_targets:
        val_mean, val_std = valences.mean(), valences.std()
        aro_mean, aro_std = arousals.mean(), arousals.std()

        valences = (valences - val_mean) / (val_std + 1e-8)
        arousals = (arousals - aro_mean) / (aro_std + 1e-8)

        target_stats = {
            "valence": {"mean": val_mean, "std": val_std},
            "arousal": {"mean": aro_mean, "std": aro_std}
        }
           
        print("\n Target normalization applied:")
        print(f"   Valence → mean={val_mean:.4f}, std={val_std:.4f}")
        print(f"   Arousal → mean={aro_mean:.4f}, std={aro_std:.4f}")
    else:
        print("\n No normalization applied")


    # Build dictionary for easy access
    data_dict = {
        "waveforms": [d["waveform"] for d in data],
        "valences": [float(d["valence"]) for d in data],
        "arousals": [float(d["arousal"]) for d in data],
        "filenames": [d["filename"] for d in data],
        "sample_rate": target_sr,
    }

    if target_stats is not None:
        data_dict["target_stats"] = target_stats

    print("\nData ready as a Torch object")
    print(f"Total samples: {len(data_dict['waveforms'])}\n")

    return data_dict