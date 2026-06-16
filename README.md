# Music Genre Classification

[](https://github.com/KuroKami2023/music-genre-classification)
[](LICENSE)
[](https://www.python.org/)
[](https://scikit-learn.org/)
[](https://librosa.org/)
[]()

> Classify music into 10 genres by extracting audio features with librosa and training ensemble classifiers with hyperparameter tuning.

---

> 💡 **Portfolio demo:** A simplified browser-based version is available in the [portfolio website](https://kurokami2023.github.io).

---

## Features

- [x] **Multi-Model Classification** — SVM, Random Forest, KNN, and MLP with GridSearchCV tuning
- [x] **Comprehensive Feature Extraction** — MFCC, Chroma, Spectral, Tonnetz, Mel-spectrogram (88 features)
- [x] **Parallel Processing** — Multi-core feature extraction for large audio datasets
- [x] **Interactive CLI Prediction** — Predict genre for any .wav file with confidence probabilities
- [x] **Rich Visualizations** — PCA/t-SNE embeddings, MFCC heatmaps, correlation analysis
- [x] **Model Persistence** — Save/load trained models and scalers via Joblib
- [x] **Feature Space Exploration** — Dimensionality reduction visualization for genre separability

## Dataset: GTZAN

| Property | Value |
|----------|-------|
| **Source** | [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| **Genres** | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Samples** | 100 per genre (1,000 total) |
| **Duration** | 30 seconds each |
| **Format** | .wav, 22,050 Hz, mono |

## Feature Extraction (librosa)

| Feature Group | Features | Count |
|--------------|----------|-------|
| **MFCC** | 20 coefficients × (mean, std) | 40 |
| **Chroma STFT** | 12 bins × (mean, std) | 24 |
| **Spectral Centroid** | mean, std | 2 |
| **Spectral Bandwidth** | mean, std | 2 |
| **Spectral Rolloff** | mean, std | 2 |
| **Zero-Crossing Rate** | mean, std | 2 |
| **RMS Energy** | mean, std | 2 |
| **Tonnetz** | 6 components × (mean, std) | 12 |
| **Mel-Spectrogram** | mean, std | 2 |
| **Total** | | **88 features** |

## Model Architecture / Approach

### Pipeline

```
Audio Files → librosa Feature Extraction → Feature Matrix (88 dims)
→ Train/Test Split (80/20, stratified) → StandardScaler
→ GridSearchCV (5-fold) → Train 4 Classifiers → Evaluate & Compare
```

### Algorithms

| Model | Search Space |
|-------|-------------|
| **SVM (RBF)** | C ∈ {0.1, 1, 10}, gamma ∈ {scale, auto, 0.01} |
| **Random Forest** | n_estimators ∈ {100, 200, 300}, max_depth ∈ {10, 20, None} |
| **KNN** | k ∈ {3, 5, 7, 9}, weights ∈ {uniform, distance} |
| **MLP** | hidden layers ∈ [(128,64), (256,128,64), (64,32)], learning rate ∈ {1e-3, 1e-4} |

## Results

| Model | Accuracy | F1-Score (Weighted) | Precision | Recall |
|-------|----------|---------------------|-----------|--------|
| SVM (RBF) | ~71% | ~0.70 | ~0.72 | ~0.71 |
| Random Forest | ~68% | ~0.67 | ~0.69 | ~0.68 |
| KNN | ~62% | ~0.61 | ~0.63 | ~0.62 |
| MLP | ~69% | ~0.68 | ~0.70 | ~0.69 |

*Best performer: SVM with RBF kernel — consistent F1 across all genres.*

# Tech Stack

- **Python** — Core language
- **Librosa** — Audio analysis and feature extraction
- **Scikit-learn** — ML models, GridSearchCV, metrics
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Publication-quality visualizations
- **Joblib** — Model persistence
- **SoundFile** — Audio file I/O

## Installation

```bash
# Clone the repository
git clone https://github.com/KuroKami2023/music-genre-classification.git
cd music-genre-classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Extract Features

```bash
python extract_features.py data/genres features.csv
```

Extracts 88 audio features from .wav files using parallel processing. Saves to CSV.

### 2. Train Classifiers

```bash
python train_classifier.py features.csv
```

Runs GridSearchCV for all 4 models, saves best models to `models/`, and generates confusion matrices.

### 3. Visualize Feature Space

```bash
python visualize_features.py features.csv
```

Generates MFCC heatmaps, PCA/t-SNE plots, feature distributions, and correlation analysis.

### 4. Predict Genre for New Audio

```bash
python predict_genre.py path/to/song.wav
python predict_genre.py path/to/song.wav --model random_forest
```

Returns predicted genre with confidence and full probability distribution.

## Project Structure

```
music-genre-classification/
├── extract_features.py       # librosa feature extraction with multiprocessing
├── train_classifier.py       # GridSearchCV training for SVM, RF, KNN, MLP
├── predict_genre.py          # CLI prediction with probability breakdown
├── visualize_features.py     # EDA: PCA/t-SNE, MFCC heatmaps, correlation
├── data/
│   └── README.md             # Dataset download instructions
├── results/                  # Generated plots (created after training)
├── models/                   # Saved models and artifacts (created after training)
├── requirements.txt
├── LICENSE
└── README.md
```

## Training Details & Hyperparameters

| Parameter | Value |
|-----------|-------|
| Test split | 20% (stratified) |
| Cross-validation folds | 5 |
| Scaling | StandardScaler |
| Random state | 42 |
| Grid search verbosity | 1 |
| MLP max iterations | 500 (with early stopping) |
| SVM probability | True (for predict_proba) |

### Best Found Parameters (Typical)

| Model | Best Params |
|-------|------------|
| SVM | `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}` |
| Random Forest | `{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}` |
| KNN | `{'n_neighbors': 5, 'p': 2, 'weights': 'distance'}` |
| MLP | `{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (256, 128, 64), 'learning_rate_init': 0.001}` |

## Performance Benchmarks

| Stage | Dataset (1,000 files) | Dataset (10,000 files) |
|-------|----------------------|------------------------|
| Feature extraction | ~30s | ~5 min |
| SVM training | ~45s | ~8 min |
| RF training | ~30s | ~3 min |
| KNN training | ~2s | ~15s |
| MLP training | ~60s | ~10 min |
| Single prediction | < 1s | < 1s |

## Future Improvements

- [ ] Deep learning with spectrogram CNNs (VGGish, ResNet)
- [ ] Transfer learning with OpenL3 or YAMNet embeddings
- [ ] Real-time genre classification from microphone input
- [ ] Music recommendation system based on genre similarity
- [ ] Multi-label classification for genre mixtures
- [ ] Audio segmentation for track-level vs. segment-level classification
- [ ] REST API deployment with FastAPI

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
