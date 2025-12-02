# 🎵 Amazon Music Clustering Dashboard  

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow?logo=plotly)  
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)  

---

## 📌 Project Overview  
This project analyzes an **Amazon Music dataset (single genre artists)** and applies **K-Means clustering** to group songs based on audio features.  
An interactive **Streamlit dashboard** is provided for clustering, visualization, and downloading results.  

---

## 📂 Project Structure  
```
├── Preprocess.ipynb         # Jupyter notebook for preprocessing steps
├── single_genre_artists.csv # Dataset used for clustering
├── stream.py                # Streamlit app (dashboard)
├── require.txt              # Required Python libraries
```

---

## ⭐ Key Features
### 🔍 1. Complete ML Pipeline

The script performs:

Basic EDA (info, missing values, feature distributions)

Preprocessing

Scaling using StandardScaler or MinMaxScaler

PCA (2D visualization)

KMeans clustering

DBSCAN clustering

Hierarchical clustering (Ward linkage)

Inertia curve (Elbow Method)

Silhouette Score & Davies–Bouldin Index

Cluster summary statistics export

Visualizations saved automatically

### 📊 2. Interactive Streamlit Dashboard

The dashboard allows users to:

Choose number of clusters (k)

Run KMeans instantly

Visualize cluster distribution

View PCA scatter plot

Inspect cluster-wise feature averages

Explore correlation heatmaps

Read cluster-wise music interpretation summaries
---

## 🛠️ Installation  

1. **Clone repository / download project files**  
   ```bash
   git clone [<(https://github.com/SUDHAKARMVRS/Amazon_Music_Clustering.git)>]
   cd project-folder
   ```


3. **Install dependencies**  
   ```bash
   pip install -r require.txt
   ```

---

## ▶️ Run the Dashboard  
```bash
streamlit run stream.py
```

After running, open 👉 `http://localhost:8501` in your browser.  

---

## 📊 Dataset Columns  
| Column          | Description |
|-----------------|-------------|
| `danceability`  | Suitability for dancing (0–1) |
| `energy`        | Intensity and activity (0–1) |
| `loudness`      | Loudness in dB |
| `speechiness`   | Spoken word presence (0–1) |
| `acousticness`  | Acoustic probability (0–1) |
| `instrumentalness` | Instrumental probability (0–1) |
| `liveness`      | Live performance probability (0–1) |
| `valence`       | Positivity of mood (0–1) |
| `tempo`         | Beats per minute |
| `duration_ms`   | Song duration in ms |

---
## 📈 Evaluation Metrics
### For KMeans:

Sum of Squared Errors (SSE/Inertia)

Silhouette Score

Davies–Bouldin Index

### For DBSCAN:

Silhouette Score (if clusters > 1)

Number of clusters

Noise points detected


## 📎 Requirements  
See `require.txt`:  
```
pandas
streamlit
matplotlib
seaborn
numpy
```
---
## 🧑‍💻 Author  
### Sudhakar M
📧sudhakar.mvrs@gmail.com| 🌐 (https://www.linkedin.com/in/sudhakar-m-657ba787/)
## 🛠️ Used Tools
🐍 Python | 🚀 Streamlit | 🤖 Scikit-learn | 📊 Pandas & Matplotlib  
