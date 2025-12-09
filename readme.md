# 🎵 Amazon Music Clustering Project  

### *Unsupervised Machine Learning + Streamlit Dashboard*

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow?logo=plotly)  
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)  
---

## ✨ Overview   
This project performs **end-to-end clustering** unsupervised machine learning project featuring clustering, PCA visualizations, evaluation metrics,and a modern interactive Streamlit dashboard.

It includes:

- Data Cleaning  
- Feature Engineering  
- Scaling (StandardScaler / MinMaxScaler)  
- PCA Dimensionality Reduction  
- KMeans, DBSCAN, Hierarchical Clustering  
- Evaluation (Silhouette, DB Index)  
- Auto-generated visualizations  
- Streamlit Dashboard  

---

## 📂 Project Structure  

```
Amazon-Music-Clustering/
│
├── complete.py      # Full ML pipeline script
├── stream.py          # Streamlit dashboard app
├── dataset.csv           # Input dataset
├── outputs/              # Auto-generated results
│     ├── elbow.png
│     ├── silhouette.png
│     ├── pca_plot.png
│     ├── dendrogram.png
│     └── clustered_data.csv
└── README.md
```

---

## 🚀 Features  

### 🔍 Machine Learning  
- Complete preprocessing workflow  
- PCA visualization  
- KMeans clustering with automated metrics  
- DBSCAN cluster detection  
- Hierarchical clustering dendrogram  
- Evaluation using Silhouette & DB Score  

### 📊 Streamlit Dashboard  
- Choose number of clusters  
- Interactive PCA visualization  
- Heatmaps & distributions
- Cluster insights
---

## ▶️ How to Run  

### **Install dependencies**  
```
pip install -r require.txt
```

### **Run the ML Pipeline**  
```
python complete.py
```

### **Launch the Dashboard**  
```
python -m streamlit run stream.py
```

---

## 📁 Outputs Generated  

| Output File | Description |
|-------------|-------------|
| **elbow.png** | Optimal K visualization |
| **silhouette.png** | Silhouette score plot |
| **pca_plot.png** | PCA 2D scatter plot |
| **clustered_data.csv** | Songs with cluster labels |
| **dendrogram.png** | Hierarchical clustering tree |

---

## 🧠 Example Cluster Interpretations  
- **Cluster 0:** Energetic, high danceability songs  
- **Cluster 1:** Acoustic & calm tracks  
- **Cluster 2:** Rap / spoken word heavy  
- **Cluster 3:** Instrumental / low vocal presence  

---

## 🛠️ Used Tools

🐍 Python | 🚀 Streamlit | 🤖 Scikit-learn | 📊 Pandas & Matplotlib  

---

## 👨‍💻 Author  
### Sudhakar M
📧sudhakar.mvrs@gmail.com| 🌐 (https://www.linkedin.com/in/sudhakar-m-657ba787/)


---

