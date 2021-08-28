import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as cluster_algorithm
from sklearn.cluster import AgglomerativeClustering

shopping_data = pd.read_csv('shopping_data.csv')
print(shopping_data)
data = shopping_data.iloc[:, 3:5].values
# Annual Income, Spending Score
print(data.shape)

plt.figure(figsize=(10, 7))
# size in inches width and height for pop up window
plt.title("Market  Segmentation Dendrogram")

dendrogram = cluster_algorithm.dendrogram(cluster_algorithm.linkage(data, method='ward'))
# linkage is constructing linkage matrix
# method
# can find optimal number of cluster for clustering algorithm
# we can find out 5 cluster is optimal number for clustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
plt.title('Market Segmentation')
plt.xlabel('Annual income')
plt.ylabel('Spending Score')

plt.show()
