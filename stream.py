import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="🎵 Amazon Music Clustering", layout="wide")

# -------------------------------------------------------
# CUSTOM CSS — GLOWING SIDEBAR, GLITTER TEXT, RAINBOW TABS
# -------------------------------------------------------
st.markdown("""
<style>
/* -------------------------
   Plain sidebar with soft glow
---------------------------*/
[data-testid="stSidebar"] {
    background: #2c3e50;  /* solid color */
    border-radius: 10px;
    box-shadow: 0 0 15px 3px rgba(76, 175, 255, 0.6); /* soft blue glow */
    padding-top: 25px;
    transition: all 0.3s ease;
}

/* Sidebar title */
.sidebar-title {
    font-size: 22px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
}

/* Sidebar labels and sliders */
.stSlider label, .stSelectbox label {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 15px !important;
}

/* Sidebar buttons */
.stButton>button {
    background: #4f6ef7;
    color: white;
    border-radius: 10px;
    padding: 8px 14px;
    width: 100%;
    font-size: 16px;
    border: none;
    transition: 0.25s ease;
}

.stButton>button:hover {
    background: #3d57d6;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* -------------------------
   Glowing Sidebar
---------------------------*/
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #ff4d4d, #ff94a5, #ffcc70);
    padding-top: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px 5px #ff4d4d;
    transition: 0.5s all;
}
[data-testid="stSidebar"]:hover {
    box-shadow: 0 0 40px 10px #ff94a5;
}

/* Sidebar title */
.sidebar-title {
    font-size: 22px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff7fe6, #7d5bff, #4ecbff, #ff7fe6);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glitter 4s infinite linear;
}

/* Glitter animation */
@keyframes glitter {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glitter text for H1 */
h1 {
    font-weight: 900 !important;
    font-size: 36px !important;
    background: linear-gradient(90deg, #ff6ad5, #6a85ff, #4ff0ff, #ff6ad5);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glitter 4s linear infinite;
}

/* Tabs base */
.stTabs [data-baseweb="tab"] {
    position: relative;
    padding: 10px 18px;
    border-radius: 12px;
    background: white;
    border: 1px solid #e0e0e0;
    transition: 0.3s ease;
}

/* Glitter text in tabs */
.stTabs [data-baseweb="tab"] p {
    background: linear-gradient(90deg, #ff6bd5, #8f5aff, #4eceff, #ff6bd5);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glitter 5s linear infinite;
    font-weight: 800;
}

/* Active Tab Rainbow Border */
@keyframes rainbowBorder {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stTabs [data-baseweb="tab"][aria-selected="true"]::before {
    content: "";
    position: absolute;
    inset: 0;
    padding: 3px;
    border-radius: 12px;
    background: linear-gradient(
        90deg,
        #ff0000,#ff7a00,#ffee00,#33ff00,#00ffee,#0066ff,#cc00ff,#ff00aa,#ff0000
    );
    background-size: 400% 400%;
    animation: rainbowBorder 5s linear infinite;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
}

/* Buttons inside sidebar */
.stButton>button {
    background: #4f6ef7;
    color: white;
    border-radius: 10px;
    padding: 8px 14px;
    width: 100%;
    font-size: 16px;
    border: none;
    transition: 0.25s ease;
}
.stButton>button:hover {
    background: #3d57d6;
}

/* Sidebar labels */
.stSlider label, .stSelectbox label {
    color: #fff !important;
    font-weight: 600;
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------
st.markdown('<h1 class="glitter-text">🎵 Amazon Music Clustering Dashboard</h1>', unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
k = st.sidebar.slider("🔢 Number of clusters (k)", 2, 10, 5, step=1)
run_clustering = st.sidebar.button("🚀 Run")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("single_genre_artists.csv")

df = load_data()

features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# PREPROCESS (EXACT SAME AS NOTEBOOK)
# ---------------------------------------
FEATURES = [
 "danceability", "energy", "loudness", "speechiness","acousticness",
 "instrumentalness", "liveness","valence", "tempo", "duration_ms"
]

# Keep only features
X = df[FEATURES].copy()

# Fill missing with median (same as notebook)
for col in X.columns:
    X[col].fillna(X[col].median(), inplace=True)

# Log transform only for duration
if "duration_ms" in X.columns:
    X["duration_ms"] = np.log1p(X["duration_ms"])

# StandardScaler (same as notebook)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (same as notebook)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Assign for plotting
df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]

explained = pca.explained_variance_ratio_

# -------------------------------------------------------
# GLOBAL PCA (Do it once and use everywhere)
# -------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]

explained = pca.explained_variance_ratio_

pc1 = explained[0]
pc2 = explained[1]
total = pc1 + pc2


# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 CLUSTER DISTRIBUTION",
    "🎨 PCA SCATTER PLOT",
    "📄 FEATURES",
    "🔍 CORRELATION",
    "📝 SUMMARY"
])

# -------------------------------------------------------
# TAB 1 – CLUSTER DISTRIBUTION
# -------------------------------------------------------
with tab1:
    if run_clustering:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        st.subheader("📊 Cluster Distribution")
        cluster_counts = df["cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["cluster", "count"]

        fig = px.bar(
            cluster_counts,
            x="cluster",
            y="count",
            color="count",
            labels={"cluster": "Cluster", "count": "Songs"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
    
        st.info("👈 Select Cluster count and click **Run Clustering**")


# -------------------------------------------------------
# TAB 2 – PCA SCATTER PLOT
# -------------------------------------------------------
with tab2:
    if not run_clustering:
        st.warning("⚠ Run clustering first.")
        st.stop()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(df["pca1"], df["pca2"],
                         c=df["cluster"], cmap="Set2", alpha=0.7)
    ax.set_title("PCA Visualization")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)


# PCA Explained Variance Table


st.markdown("### 📊 PCA Explained Variance Summary")

pca_table = pd.DataFrame({
    "Component": ["PC1", "PC2", "Total (PC1+PC2)"],
    "Explained Variance": [
        f"{pc1:.4f} ({pc1*100:.2f}%)",
        f"{pc2:.4f} ({pc2*100:.2f}%)",
        f"{total:.4f} ({total*100:.2f}%)",
    ],
    "Interpretation": [
        "Captures major variation (energy, loudness, tempo)",
        "Mood/style variation (danceability, valence)",
        "Information retained after PCA reduction"
    ]
})

st.dataframe(pca_table)


# Interpretation Box
st.info("""
**What does this mean?**
- PCA 2D plot retains **48.59%** of musical information  
- Enough for visualizing clusters  
- Shows structural patterns clearly, even after reduction
""")


# -------------------------------------------------------
# TAB 3 – CLUSTER FEATURES
# -------------------------------------------------------
with tab3:
    if not run_clustering:
        st.warning("⚠ Run clustering first.")
        st.stop()

    st.subheader("📌 Mean Feature Values for Each Cluster")
    cluster_profile = df.groupby("cluster")[features].mean()
    st.dataframe(cluster_profile.style.highlight_max(axis=0))

# -------------------------------------------------------
# TAB 4 – CORRELATION HEATMAP
# -------------------------------------------------------
with tab4:
    if not run_clustering:
        st.warning("⚠ Run clustering first.")
        st.stop()

    st.subheader("🔍 Correlation Heatmap")
    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# TAB 5 – CLUSTER SUMMARY
# -------------------------------------------------------
with tab5:
    if not run_clustering:
        st.warning("⚠ Run clustering first.")
        st.stop()

    st.subheader("📝 Cluster Report")
    cluster_profile = df.groupby("cluster")[features].mean()

    for c, row in cluster_profile.iterrows():
        desc = f"### 🎯 C - {c}\n"
        if row['danceability'] > 0.65: desc += "- **Party Songs**\n"
        if row['energy'] > 0.6: desc += " - **Workout / High-Intensity Songs**\n"
        if row['acousticness'] > 0.6: desc += "- **Chill Acoustic / Calm Songs**\n"
        if row['instrumentalness'] > 0.6: desc += "-**Instrumental / BGM's**\n"
        if row['speechiness'] > 0.6: desc += "- **Spoken / Rap / Hip-hop**\n"
        if row['valence'] > 0.6: desc += "- **Feel-Good / Happy Songs**\n"
        if row['tempo'] > 120: desc += "- **Upbeat / Fast Songs**\n"
        st.markdown(desc)
