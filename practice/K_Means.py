# import numpy as np

# # class KMeans:
# #     def __init__(self, n_clusters=3, max_iters=100):
# #         self.n_clusters = n_clusters
# #         self.max_iters = max_iters

# #     def fit(self, X):
# #         #randomly select 4 points from the dataset
# #         self.centroids = X[np.random.choice(X.shape[0], self.n_clusters)]

# #         for _ in range(self.max_iters):
# #             # Assign clusters based on closest centroid
# #             self.labels = np.array([self._closest_centroid(x) for x in X])

# #             # Update centroids by calculating the mean of points in each cluster
# #             self.centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

# #     def _closest_centroid(self, x):
# #         distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
# #         return np.argmin(distances)

# #     def _euclidean_distance(self, x1, x2):
# #         return np.sqrt(np.sum((x1 - x2) ** 2))

# #     def predict(self, X):
# #         return np.array([self._closest_centroid(x) for x in X])


# class KMeans:
#     def __init__(self,k=3,max_iters=10):
#         self.k = k
#         self.max_iters = max_iters

#     def fit(self,x):
#         self.centroids = x[np.random.choice(x.shape[0],self.k)]

#         for _ in range(self.max_iters):
#             self.labels = np.array([self._closest_centroid(ele) for ele in x])

#             self.centroids = np.array([x[self.labels == i].mean(axis=0) for i in range(self.k)])
#             print("done")

#     def _closest_centroid(self,x):
#         distances = [self._euclidean_distance(x,centroid) for centroid in self.centroids]
#         return np.argmin(distances)
    
#     def _euclidean_distance(self,x,centroid):
#         return np.sqrt(np.sum((x-centroid)**2))
    
#     def predict(self,x):
#         return np.array([self._closest_centroid(ele) for ele in x])
    




# # Example usage
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# # Generate sample data
# X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# # Fit KMeans
# kmeans = KMeans(k=4)
# kmeans.fit(X)

# # Plot the clusters and centroids
# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', marker='o', edgecolor='k')
# plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X', label="Centroids")
# plt.legend()
# plt.show()


# import pandas as pd

# # Creating a DataFrame with the provided data
# data = {
#     'points': [18.0, 19.0, 14.0, 14.0, 11.0, 20.0, 28.0, 30.0, 31.0, 35.0, 
#                33.0, 25.0, 25.0, 27.0, 29.0, 30.0, 19.0],
#     'assists': [3.0, 4.0, 5.0, 4.0, 7.0, 8.0, 7.0, 6.0, 9.0, 12.0, 
#                 14.0, 9.0, 4.0, 3.0, 4.0, 12.0, 15.0],
#     'rebounds': [15, 14, 10, 8, 14, 13, 9, 5, 4, 11, 
#                  6, 5, 3, 8, 12, 7, 6]
# }

# df = pd.DataFrame(data)
# print(df)

# import numpy as np
# import matplotlib.pyplot as plt

# class KMeans:
#     def __init__(self,k=3,max_iters=10):
#         self.k = k
#         self.max_iters = max_iters

#     def fit(self,x):
#         self.centroids = x[np.random.choice(x.shape[0],self.k,replace=False)]

#         for _ in range(self.max_iters):
#             self.labels = np.array([self._closest_centroid(ele) for ele in x])

#             self.centroids = np.array([x[self.labels == i].mean(axis=0) for i in range(self.k)])
#             # print("done")

#     def _closest_centroid(self,x):
#         distances = [self._euclidean_distance(x,centroid) for centroid in self.centroids]
#         return np.argmin(distances)
    
#     def _euclidean_distance(self,x,centroid):
#         return np.sqrt(np.sum((x-centroid)**2))
    
#     def predict(self,x):
#         return np.array([self._closest_centroid(ele) for ele in x])
    

#     def calculate_sse(self, X):
#         sse = 0
#         print(self.labels)
#         for i in range(self.k):
#             cluster_points = X[self.labels == i]
#             if len(cluster_points) > 0:  # Ensure there are points in the cluster
#                 sse += np.sum((cluster_points - self.centroids[i]) ** 2)
#         return sse
# X = df[['points', 'assists', 'rebounds']].values

# sse_values = []
# k_values = range(1, 11)

# for k in k_values:
#     kmeans = KMeans(k=k)
#     kmeans.fit(X)
#     sse = kmeans.calculate_sse(X)
#     print(sse)
#     sse_values.append(sse)

# # # Plotting SSE vs. Number of Clusters
# plt.plot(k_values, sse_values, marker='o')
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Sum of Squared Errors (SSE)")
# plt.title("Elbow Method for Optimal k")
# plt.show()

# # Set the optimal k value (you can set it based on the elbow plot, e.g., k = 3)
# optimal_k = 3
# kmeans = KMeans(k=optimal_k)
# kmeans.fit(X)
# predictions = kmeans.predict(X)

# # Plotting the clusters
# for i in range(optimal_k):
#     cluster_points = X[predictions == i]
#     print(cluster_points)
#     print("************")
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
    
# # Plot centroids
# plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='black', marker='x', s=100, label='Centroids')
# plt.xlabel("Points")
# plt.ylabel("Assists")
# plt.title("K-means Clustering of Basketball Players")
# plt.legend()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'points': [18.0, 19.0, 14.0, 14.0, 11.0, 20.0, 28.0, 30.0, 31.0, 35.0, 
               33.0, 25.0, 25.0, 27.0, 29.0, 30.0, 19.0],
    'assists': [3.0, 4.0, 5.0, 4.0, 7.0, 8.0, 7.0, 6.0, 9.0, 12.0, 
                14.0, 9.0, 4.0, 3.0, 4.0, 12.0, 15.0],
    'rebounds': [15, 14, 10, 8, 14, 13, 9, 5, 4, 11, 
                 6, 5, 3, 8, 12, 7, 6]
}

df = pd.DataFrame(data)
print(df)

class KMeans:
    def __init__(self,k=3,max_iter=10):
        self.k=k
        self.max_iter=max_iter

    def fit(self,x):    
        x=x.values
        self.centroids = x[np.random.choice(x.shape[0], self.k, replace=False)]

        for _ in range(self.max_iter):
            self.labels = np.array([self._get_closest_centroid(ele) for ele in x])

            self.centroids = np.array([x[self.labels == k].mean(axis=0) for k in range(self.k)])

    def _get_closest_centroid(self,ele):
        distances = [self.euclidean_distance(ele,cen) for cen in self.centroids ]
        return np.argmin(distances)
    
    def euclidean_distance(self, ele,cen):
        return np.sqrt(np.sum((ele-cen)**2))

    def predict(self,x):
        x=x.values
        return np.array([self._get_closest_centroid(ele) for ele in x])
    
    def sse(self,x):
        x=x.values
        sse=0
        for i in range(self.k):
            cluster_points = x[self.labels == i]
            if len(cluster_points)>0:
                sse+=np.sum((cluster_points-self.centroids[i])**2)   
        return sse



k_range = range(1,11)
sse_arr=[]

for k in k_range:
    km = KMeans(k=k)
    km.fit(df)
    sse = km.sse(df)
    sse_arr.append(sse)

plt.plot(k_range,sse_arr,marker='o')
plt.show()

optimal_k = 3
opt_k = KMeans(k=optimal_k)
opt_k.fit(df)
predicitons = opt_k.predict(df)

x = df.values
print(x)
for i in range(optimal_k):
    cluster_points = x[predicitons==i]
    plt.scatter(cluster_points[:,0],cluster_points[:,1],label = f'Cluster {i+1}' )

plt.scatter(opt_k.centroids[:,0],opt_k.centroids[:,1],marker='x',color='black')
plt.show()