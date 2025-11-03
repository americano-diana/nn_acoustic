"""
Script to extract only .wav files from the wav_files directory and resample them to 16kHz.
"""

import os
from pathlib import Path
import torchaudio
from tqdm import tqdm

def clean_wavs(input_dir, output_dir, target_sr=16000):
    """
    Extracts all .wav files from input_dir, resamples them to target_sr, and saves them in output_dir.

    Args:
        input_dir (str | Path): Directory containing original wav files.
        output_dir (str | Path): Directory to save resampled wavs.
        target_sr (int): Target sample rate (default 16kHz).
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = [f for f in input_dir.glob("*.wav")]
    print(f"Found {len(wav_files)} .wav files in {input_dir}")

    for wav_file in tqdm(wav_files, desc="Resampling WAVs"):
        try:
            waveform, sr = torchaudio.load(wav_file)

            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            # Save resampled file
            output_path = output_dir / wav_file.name
            torchaudio.save(output_path, waveform, target_sr)

        except Exception as e:
            print(f" Error processing {wav_file.name}: {e}")

    print(f"\n Resampled files saved in: {output_dir}\n")

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]  # project root
    input_dir = root / "data" / "raw" / "wav_files"
    output_dir = root / "data" / "processed" / "wav_16k"

    clean_wavs(input_dir, output_dir)