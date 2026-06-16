"""
Extract audio features using librosa for music genre classification.

Features extracted:
- MFCC (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral centroid, bandwidth, rolloff
- Zero-crossing rate
- RMS energy
- Mel-spectrogram
- Tonnetz
"""

import os
import numpy as np
import pandas as pd
import librosa
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def extract_features(audio_path: str, sr: int = 22050, n_mfcc: int = 20) -> Dict[str, float]:
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=30)
    except Exception as e:
        print(f'Error loading {audio_path}: {e}')
        return {}

    if len(y) == 0:
        return {}

    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i}_std'] = np.std(chroma[i])

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_std'] = np.std(zcr)

    rms = librosa.feature.rms(y=y)[0]
    features['rms_energy_mean'] = np.mean(rms)
    features['rms_energy_std'] = np.std(rms)

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(tonnetz.shape[0]):
        features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
        features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spec_mean'] = np.mean(mel_spec_db)
    features['mel_spec_std'] = np.std(mel_spec_db)

    return features


def extract_all_features(data_dir: str, out_csv: str = 'features.csv') -> pd.DataFrame:
    genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f'Found genres: {genres}')

    all_features = []
    total_files = 0
    file_paths = []

    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        for fname in os.listdir(genre_dir):
            if fname.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_paths.append((os.path.join(genre_dir, fname), genre))
                total_files += 1

    print(f'Found {total_files} audio files')
    print('Extracting features...')

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(_extract_single, file_paths), total=len(file_paths)))

    for (path, genre), features in zip(file_paths, results):
        if features:
            features['filename'] = os.path.basename(path)
            features['genre'] = genre
            all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv(out_csv, index=False)
    print(f'Features saved to {out_csv}')
    print(f'Feature matrix shape: {df.shape}')
    return df


def _extract_single(args):
    path, genre = args
    return extract_features(path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python extract_features.py <data_directory> [output_csv]')
        sys.exit(1)
    data_dir = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else 'features.csv'
    extract_all_features(data_dir, out_csv)
