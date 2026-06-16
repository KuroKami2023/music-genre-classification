"""
Train classifiers for music genre classification.
Supports SVM, Random Forest, KNN, and MLP with hyperparameter tuning via GridSearchCV.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
import joblib
from typing import Dict, Tuple, Any

RESULTS_DIR = 'results'
MODELS_DIR = 'models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_features(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    df = pd.read_csv(csv_path)
    label_col = 'genre'
    feature_cols = [c for c in df.columns if c not in ['filename', 'genre']]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    return df, X, y, le


def get_classifiers() -> Dict[str, Tuple[Any, Dict]]:
    return {
        'SVM': (
            SVC(kernel='rbf', random_state=42, probability=True),
            {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01],
                'kernel': ['rbf'],
            }
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
            }
        ),
        'KNN': (
            KNeighborsClassifier(n_jobs=-1),
            {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
            }
        ),
        'MLP': (
            MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
            {
                'hidden_layer_sizes': [(128, 64), (256, 128, 64), (64, 32)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.0001],
                'alpha': [0.0001, 0.001],
            }
        ),
    }


def train_and_evaluate(
    X_train, X_test, y_train, y_test, class_names
) -> Dict[str, Dict[str, Any]]:
    classifiers = get_classifiers()
    all_results = {}

    for name, (clf, param_grid) in classifiers.items():
        print(f'\n{"="*60}')
        print(f'Training {name} with GridSearchCV...')
        print(f'{"="*60}')
        grid = GridSearchCV(
            clf, param_grid, cv=5, scoring='f1_weighted',
            n_jobs=-1, verbose=1, return_train_score=True
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_scores = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f'\nBest parameters: {grid.best_params_}')
        print(f'Best CV score: {grid.best_score_:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')
        print(f'Weighted F1: {f1:.4f}')

        model_path = os.path.join(MODELS_DIR, f'{name.replace(" ", "_").lower()}_model.pkl')
        joblib.dump(best_model, model_path)
        print(f'Model saved to {model_path}')

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name} - Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_").lower()}.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f'Confusion matrix saved to {cm_path}')

        all_results[name] = {
            'best_params': grid.best_params_,
            'cv_score': grid.best_score_,
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': best_model,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred, target_names=class_names, digits=4),
        }

    return all_results


def compare_models(all_results: Dict[str, Dict], class_names: list):
    summary_data = []
    for name, res in all_results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': f'{res["test_accuracy"]:.4f}',
            'Precision': f'{res["precision"]:.4f}',
            'Recall': f'{res["recall"]:.4f}',
            'F1-Score': f'{res["f1_score"]:.4f}',
            'CV Score': f'{res["cv_score"]:.4f}',
        })
    summary_df = pd.DataFrame(summary_data)
    print('\n' + '=' * 60)
    print('Model Comparison Summary')
    print('=' * 60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    print(f'Summary saved to {RESULTS_DIR}/model_comparison.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_results))
    width = 0.2
    metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
    for i, metric in enumerate(metrics):
        values = [all_results[name][metric] for name in all_results]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(list(all_results.keys()))
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f'Comparison chart saved to {RESULTS_DIR}/model_comparison.png')


def main():
    if len(sys.argv) < 2:
        csv_path = 'features.csv'
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f'Error: Features file not found at {csv_path}')
        print('Run extract_features.py first to generate features.')
        sys.exit(1)

    print('=' * 60)
    print('Music Genre Classification - Model Training')
    print('=' * 60)
    df, X, y, label_encoder = load_features(csv_path)
    print(f'Loaded {len(df)} samples with {X.shape[1]} features')
    print(f'Genres: {list(label_encoder.classes_)}')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    print(f'Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}')
    all_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, label_encoder.classes_)
    compare_models(all_results, label_encoder.classes_)
    best_model_name = max(all_results, key=lambda k: all_results[k]['f1_score'])
    print(f'\nBest model: {best_model_name} (F1={all_results[best_model_name]["f1_score"]:.4f})')


if __name__ == '__main__':
    main()
