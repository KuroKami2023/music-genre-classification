"""
Predict music genre for a new audio file using trained models.
"""

import os
import sys
import numpy as np
import joblib
from extract_features import extract_features
from typing import Dict, List, Optional

MODELS_DIR = 'models'


def load_model_and_artifacts(model_name: str = 'svm'):
    model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')

    if not os.path.exists(model_path):
        print(f'Error: Model not found at {model_path}')
        print('Run train_classifier.py first.')
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return model, scaler, label_encoder


def predict_genre(
    audio_path: str,
    model_name: str = 'svm',
) -> Dict:
    model, scaler, label_encoder = load_model_and_artifacts(model_name)
    print(f'Extracting features from {audio_path}...')
    features = extract_features(audio_path)
    if not features:
        return {'error': 'Could not extract features from audio file'}
    feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else list(features.keys())
    feature_vector = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)
    feature_scaled = scaler.transform(feature_vector)
    probs = model.predict_proba(feature_scaled)[0] if hasattr(model, 'predict_proba') else None
    predicted_class = model.predict(feature_scaled)[0]
    genre = label_encoder.inverse_transform([predicted_class])[0]
    confidence = float(np.max(probs)) if probs is not None else 0.0
    result = {
        'filename': os.path.basename(audio_path),
        'predicted_genre': genre,
        'confidence': confidence,
    }
    if probs is not None:
        genre_probs = {}
        for i, prob in enumerate(probs):
            genre_name = label_encoder.inverse_transform([i])[0]
            genre_probs[genre_name] = float(prob)
        sorted_genres = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
        result['all_probabilities'] = sorted_genres
        result['top_3'] = sorted_genres[:3]
    return result


def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python predict_genre.py <audio_file>')
        print('  python predict_genre.py <audio_file> --model random_forest')
        sys.exit(1)
    audio_path = sys.argv[1]
    model_name = 'svm'
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model_name = sys.argv[idx + 1].lower().replace(' ', '_')
    if not os.path.exists(audio_path):
        print(f'Error: File not found: {audio_path}')
        sys.exit(1)
    result = predict_genre(audio_path, model_name)
    if 'error' in result:
        print(f'Error: {result["error"]}')
        sys.exit(1)
    print('\n' + '=' * 50)
    print(f'Filename: {result["filename"]}')
    print(f'Predicted Genre: {result["predicted_genre"]}')
    print(f'Confidence: {result["confidence"]:.2%}')
    print('=' * 50)
    if 'top_3' in result:
        print('\nTop 3 Predictions:')
        for genre, prob in result['top_3']:
            print(f'  {genre}: {prob:.2%}')
    if 'all_probabilities' in result:
        print('\nAll Probabilities:')
        for genre, prob in result['all_probabilities']:
            bar = '#' * int(prob * 50)
            print(f'  {genre:20s} |{bar:<50s}| {prob:.2%}')


if __name__ == '__main__':
    main()
