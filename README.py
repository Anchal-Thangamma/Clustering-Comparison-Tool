# Clustering-Comparison-Tool


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

st.set_page_config(page_title="Clustering Comparison Tool", layout="wide")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("Dataset Options")
dataset_type = st.sidebar.selectbox("Choose dataset", 
                                    ["Blobs", "Circles", "Moons"])

n_samples = st.sidebar.slider("Number of samples", 100, 2000, 500)

st.sidebar.title("Algorithm Parameters")
k_clusters = st.sidebar.slider("Number of Clusters (for KMeans & Agglomerative)", 2, 10, 3)
eps_val = st.sidebar.slider("DBSCAN eps", 0.1, 1.5, 0.5)
min_samples_val = st.sidebar.slider("DBSCAN min_samples", 2, 10, 5)

# ---------------------------
# Generate dataset
# ---------------------------
if dataset_type == "Blobs":
    X, y = make_blobs(n_samples=n_samples, centers=k_clusters, random_state=42)
elif dataset_type == "Circles":
    X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
else:
    X, y = make_moons(n_samples=n_samples, noise=0.07)

# ---------------------------
# Apply Algorithms
# ---------------------------

# KMeans
kmeans = KMeans(n_clusters=k_clusters)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
dbscan_labels = dbscan.fit_predict(X)

# Agglomerative
agglo = AgglomerativeClustering(n_clusters=k_clusters)
agglo_labels = agglo.fit_predict(X)

# ---------------------------
# Plot
# ---------------------------

def plot_clusters(X, labels, title):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=30)
    ax.set_title(title)
    return fig

col1, col2, col3 = st.columns(3)

with col1:
    st.pyplot(plot_clusters(X, kmeans_labels, "K-Means Clustering"))

with col2:
    st.pyplot(plot_clusters(X, dbscan_labels, "DBSCAN Clustering"))

with col3:
    st.pyplot(plot_clusters(X, agglo_labels, "Agglomerative Clustering"))
