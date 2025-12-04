# 🎵 Amazon Music Clustering Project  
### *Unsupervised Machine Learning + Streamlit Dashboard*

---

## ✨ Overview  
This project performs **end-to-end clustering** on Amazon Music track data using modern unsupervised ML techniques.  
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
├── main_pipeline.py      # Full ML pipeline script
├── dashboard.py          # Streamlit dashboard app
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
- Upload dataset  
- Choose number of clusters  
- Interactive PCA visualization  
- Cluster insights  
- Heatmaps & distributions  

---

## ▶️ How to Run  

### **Install dependencies**  
```
pip install -r requirements.txt
```

### **Run the ML Pipeline**  
```
python main_pipeline.py
```

### **Launch the Dashboard**  
```
streamlit run dashboard.py
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

## 👨‍💻 Author  
Generated for **Sudhakar M**  
by **ChatGPT (ML + Streamlit Edition)**  

---

## ⭐ Like this project?  
I can also create:  
✔ Project Report PDF  
✔ Presentation (PPT)  
✔ GitHub-ready packaging  
✔ Architecture Diagrams  

Just ask! 😊
