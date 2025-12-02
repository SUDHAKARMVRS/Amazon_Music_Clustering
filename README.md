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

## ⚙️ Features  
- 📊 **Cluster songs** using K-Means on audio features  
- 🎨 **2D PCA visualization** of clusters  
- 📌 **Cluster profiles** with average feature values  
- 📈 **Interactive controls** to select number of clusters (k)  
- ⬇️ **Export clustered dataset** as CSV  

---

## 🛠️ Installation  

1. **Clone repository / download project files**  
   ```bash
   git clone [<(https://github.com/SUDHAKARMVRS/Amazon_Music_Clustering.git)>]
   cd project-folder
   ```

2. **Create virtual environment (optional)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
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
