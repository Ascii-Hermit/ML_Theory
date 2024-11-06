import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, linkage_method='single'):
        self.linkage_method = linkage_method
        self.linkage_matrix = None

    def fit(self, X):
        # Compute the pairwise distances between points
        distance_matrix = pdist(X, metric='euclidean')
        
        # Perform hierarchical clustering using the specified linkage method
        self.linkage_matrix = linkage(distance_matrix, method=self.linkage_method)
        self.X = X  # Store data for use in SSE calculation and plotting

    def predict(self, n_clusters):
        # Cut the dendrogram to form n_clusters
        labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        return labels

    def plot_dendrogram(self):
        if self.linkage_matrix is None:
            raise ValueError("The model must be fitted before plotting the dendrogram.")
        
        plt.figure(figsize=(10, 5))
        dendrogram(self.linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        plt.show()

    def plot_clusters(self, labels):
        # Plot clusters
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels):
            cluster_points = self.X[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Cluster Visualization")
        plt.legend()
        plt.show()
    
    def calculate_sse(self, max_clusters=10):
        # Calculate and plot SSE for 1 to max_clusters
        sse = []
        for n_clusters in range(1, max_clusters + 1):
            labels = self.predict(n_clusters)
            sse_n = 0
            for label in np.unique(labels):
                cluster_points = self.X[labels == label]
                centroid = np.mean(cluster_points, axis=0)
                sse_n += np.sum((cluster_points - centroid) ** 2)
            sse.append(sse_n)
        
        # Plot SSE
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), sse, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title("SSE vs. Number of Clusters")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [8, 8], [8, 9], [25, 80]])

    # Initialize and fit the hierarchical clustering model
    model = HierarchicalClustering(linkage_method='single')
    model.fit(X)

    # Predict clusters by specifying the desired number of clusters
    labels = model.predict(n_clusters=2)
    print("Cluster Labels:", labels)

    # Plot the dendrogram
    model.plot_dendrogram()
    
    # Plot the clusters
    model.plot_clusters(labels)

    # Plot the SSE graph
    model.calculate_sse(max_clusters=5)
