# Data Source

## GTZAN Genre Collection (Recommended)

Download from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

The GTZAN dataset is the most widely used dataset for music genre classification.

### Dataset Details
- **Genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **Samples**: 100 audio files per genre (10 genres = 1000 files total)
- **Duration**: 30 seconds each
- **Format**: .wav files, 22050 Hz mono

### Instructions:
1. Download the GTZAN dataset from Kaggle
2. Extract the archive
3. Place the `genres` folder (containing genre subdirectories) in this directory:
   ```
   data/
     genres/
       blues/
       classical/
       ...
   ```
4. Extract features:
   ```bash
   python extract_features.py data/genres
   ```
5. Train classifiers:
   ```bash
   python train_classifier.py features.csv
   ```

## Alternative: TensorFlow Datasets

GTZAN is also available via `tensorflow_datasets`:
```python
import tensorflow_datasets as tfds
ds = tfds.load('gtzan', split='train')
```

## File Structure for `data/genres/`
```
genres/
  blues/
    blues.00000.wav
    blues.00001.wav
    ...
  classical/
    classical.00000.wav
    ...
  country/
  disco/
  hiphop/
  jazz/
  metal/
  pop/
  reggae/
  rock/
```
