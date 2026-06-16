"""
Visualize audio features extracted for music genre classification.
Generates MFCC heatmaps, feature distributions per genre, t-SNE/PCA plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} samples with {len(df.columns)} columns')
    return df


def plot_mfcc_heatmaps(df: pd.DataFrame, n_mfcc: int = 20, save_path: str = 'results/mfcc_heatmaps.png'):
    genres = df['genre'].unique()
    n_genres = len(genres)
    cols = min(5, n_genres)
    rows = int(np.ceil(n_genres / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten() if rows > 1 else axes
    for i, genre in enumerate(sorted(genres)):
        genre_df = df[df['genre'] == genre]
        mfcc_means = np.array([genre_df[f'mfcc_{j}_mean'].mean() for j in range(n_mfcc)])
        mfcc_stds = np.array([genre_df[f'mfcc_{j}_std'].mean() for j in range(n_mfcc)])
        ax = axes[i]
        heatmap_data = np.column_stack([mfcc_means, mfcc_stds])
        sns.heatmap(heatmap_data.T, ax=ax, cmap='coolwarm', cbar=False,
                    xticklabels=range(n_mfcc), yticklabels=['Mean', 'Std'])
        ax.set_title(genre, fontsize=10)
        ax.set_xlabel('MFCC Coefficient')
    for i in range(n_genres, len(axes)):
        axes[i].axis('off')
    plt.suptitle('MFCC Features Across Genres', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'MFCC heatmaps saved to {save_path}')


def plot_feature_distributions(df: pd.DataFrame, save_path: str = 'results/feature_distributions.png'):
    feature_cols = [c for c in df.columns if c not in ['filename', 'genre']]
    selected_features = ['spectral_centroid_mean', 'spectral_rolloff_mean',
                          'zero_crossing_rate_mean', 'rms_energy_mean']
    selected = [f for f in selected_features if f in feature_cols]
    genres = sorted(df['genre'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
    for i, feature in enumerate(selected):
        ax = axes[i]
        for j, genre in enumerate(genres):
            genre_data = df[df['genre'] == genre][feature].dropna()
            if len(genre_data) > 0:
                ax.hist(genre_data, bins=20, alpha=0.5, color=colors[j], label=genre, density=True)
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
    plt.suptitle('Feature Distributions Across Genres', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Feature distributions saved to {save_path}')


def plot_pca_tsne(df: pd.DataFrame, save_path: str = 'results/pca_tsne.png'):
    feature_cols = [c for c in df.columns if c not in ['filename', 'genre']]
    X = df[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df['genre'].values)
    genres = le.classes_
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')
    perplexity = min(30, len(df) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
    for i, genre in enumerate(genres):
        mask = y == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], color=colors[i], label=genre, alpha=0.7, s=30)
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[i], label=genre, alpha=0.7, s=30)
    ax1.set_title(f'PCA (Var: {pca.explained_variance_ratio_[0]:.2%}, {pca.explained_variance_ratio_[1]:.2%})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend(fontsize=8)
    ax2.set_title('t-SNE Visualization')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'PCA/t-SNE plot saved to {save_path}')
    return X_pca, X_tsne


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = 'results/feature_correlation.png'):
    feature_cols = [c for c in df.columns if c not in ['filename', 'genre']]
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Correlation heatmap saved to {save_path}')


def main():
    if len(sys.argv) < 2:
        csv_path = 'features.csv'
    else:
        csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f'Error: Features file not found at {csv_path}')
        print('Run extract_features.py first.')
        sys.exit(1)
    df = load_data(csv_path)
    print(f'Genres: {sorted(df["genre"].unique())}')
    plot_mfcc_heatmaps(df)
    plot_feature_distributions(df)
    plot_pca_tsne(df)
    plot_correlation_heatmap(df)
    print(f'\nAll visualizations saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
