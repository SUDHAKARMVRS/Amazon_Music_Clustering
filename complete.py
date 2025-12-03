# ------------------------------------------------------
# Amazon Music Clustering
# ------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram


# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_PATH = r"single_genre_artists.csv"  
OUTPUT_DIR = "outputs_simple"
FEATURES = ["danceability", "energy", "loudness", "speechiness","acousticness",
             "instrumentalness", "liveness","valence", "tempo", "duration_ms"]
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


# ------------------------------------------------------
# BASIC EDA
# ------------------------------------------------------
def basic_eda(df):
    print("\n--- INFO ---")
    print(df.info())

    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    # simple histograms
    for col in FEATURES:
        if col in df.columns:
            plt.figure()
            df[col].hist(bins=40)
            plt.title(f"Histogram of {col}")
            plt.savefig(f"{OUTPUT_DIR}/hist_{col}.png")
            plt.close()


# ------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------
def preprocess(df):
    df = df.copy()

    # Drop text columns if present
    drop_cols = ["track_name", "artist_name", "track_id"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Select features
    X = df[FEATURES].copy()

    # Fill missing values
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)

    # Example transform
    if "duration_ms" in X.columns:
        X["duration_ms"] = np.log1p(X["duration_ms"])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


# ------------------------------------------------------
# PCA (simple)
# ------------------------------------------------------
def do_pca(X):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    print("Explained variance (PC1, PC2):", pca.explained_variance_ratio_)

    plt.scatter(pcs[:, 0], pcs[:, 1], s=8)
    plt.title("PCA Scatter Plot")
    plt.savefig(f"{OUTPUT_DIR}/pca_scatter.png")
    plt.close()

    return pcs


# ------------------------------------------------------
# KMEANS
# ------------------------------------------------------
def kmeans_run(X):
    ks = range(2, 9)
    results = {}

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        inertia = model.inertia_
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        print(f"k={k}: inertia={inertia:.2f}, silhouette={sil:.4f}, dbi={db:4f}")

        results[k] = (sil, model, labels)

    # choose best silhouette
    best_k = max(results, key=lambda k: results[k][0])
    print("\nBest K =", best_k)

    return best_k, results[best_k][1], results[best_k][2]


# ------------------------------------------------------
# SIMPLE SCATTER PLOT
# ------------------------------------------------------
def plot_clusters(pcs, labels, name):
    plt.figure(figsize=(6, 5))
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(pcs[mask, 0], pcs[mask, 1], s=10, label=f"Cluster {c}")
    plt.legend()
    plt.title(name)
    plt.savefig(f"{OUTPUT_DIR}/{name}.png")
    plt.close()


# ------------------------------------------------------
# DBSCAN
# ------------------------------------------------------
def run_dbscan(X, pcs):
    model = DBSCAN(eps=0.6, min_samples=6)
    labels = model.fit_predict(X)

    if len(set(labels)) > 1:
        plot_clusters(pcs, labels, "DBSCAN")
        print("DBSCAN clusters:", len(set(labels)) - (1 if -1 in labels else 0))
    else:
        print("DBSCAN: No meaningful clusters found")

    return labels


# ------------------------------------------------------
# HIERARCHICAL
# ------------------------------------------------------
def run_hierarchical(X):
    # linkage for dendrogram
    sample = X[:400]
    link = linkage(sample, method="ward")

    plt.figure(figsize=(10, 6))
    dendrogram(link, truncate_mode="level", p=5)
    plt.title("Hierarchical Dendrogram")
    plt.savefig(f"{OUTPUT_DIR}/dendrogram.png")
    plt.close()


# ------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------
def run_all():
    df = load_data(DATA_PATH)

    basic_eda(df)

    clean_df, X_scaled, scaler = preprocess(df)

    pcs = do_pca(X_scaled)

    # --- KMEANS ---
    best_k, model, labels = kmeans_run(X_scaled)
    clean_df["cluster_kmeans"] = labels
    clean_df.to_csv(f"{OUTPUT_DIR}/kmeans_clusters.csv", index=False)

    plot_clusters(pcs, labels, f"kmeans_k_{best_k}")

    # --- DBSCAN ---
    run_dbscan(X_scaled, pcs)

    # --- HIERARCHICAL ---
    run_hierarchical(X_scaled)

    print("\nAll tasks complete! Check the outputs_simple/ folder.")


# ------------------------------------------------------
# RUN SCRIPT
# ------------------------------------------------------
if __name__ == "__main__":
    run_all()

