# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Load the dataset

# data = pd.read_csv(r'C:\Users\adars\OneDrive\Desktop\code\internship\Skill Intern\Iris_internship.csv')

# # Prepare the data
# X = data.iloc[:, [0, 1, 2, 3]].values

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Find the optimum number of clusters using the elbow method
# inertia = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)

# # Plot the elbow curve
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()

# # Based on the elbow curve, we choose the number of clusters (optimum) as 3

# # Apply K-means clustering with the selected number of clusters
# optimal_kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# y_kmeans = optimal_kmeans.fit_predict(X_scaled)

# # Visualize the clusters
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

# plt.scatter(optimal_kmeans.cluster_centers_[:, 0], optimal_kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
# plt.title('Cluster Visualization')
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.legend()
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset

data = pd.read_csv(r'C:\Users\adars\OneDrive\Desktop\code\internship\Skill Intern\Iris_internship.csv')
data.head()

# Prepare the data
X = data.iloc[:, [0, 1, 2, 3]].values


# Find the optimum number of clusters using the elbow method
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means clustering with the selected number of clusters
optimal_kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = optimal_kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

plt.scatter(optimal_kmeans.cluster_centers_[:, 0], optimal_kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Cluster Visualization')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

