import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# import dummy data-set > 2-D feature
x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])

plt.scatter(x[:, 0], x[:, 1], s=50)
plt.show()

linkage_matrix = linkage(x, 'single')
# optimized algorithm based on minimum spanning tree > O(N^2)
# methods : single = nearest point algorithm
# it is slow method. so unless we need dendrogram, use K-means or DBSCAN

print(linkage_matrix)
# all the information for dendrogram

dendrogram = dendrogram(linkage_matrix, truncate_mode='none')
plt.title("HC")
plt.show()

