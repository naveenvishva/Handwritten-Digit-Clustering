import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the digit dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_pca)

# Assign colors to each cluster
cluster_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF',
                  '#FF00FF', '#C0C0C0', '#800000', '#808000', '#008000']
colors = [cluster_colors[label] for label in kmeans.labels_]

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
plt.xticks([])
plt.yticks([])
plt.title("Handwritten Digit Clustering")
plt.show()
