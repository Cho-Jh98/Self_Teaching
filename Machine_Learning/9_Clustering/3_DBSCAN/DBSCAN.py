import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import datasets
import numpy as np

X, y = datasets.make_moons(n_samples=1500, noise=.05)
# makes 2 interleaving circles.
# n_sample
# shuffle or not
# noise

# print(X)  # x1, x2

x1 = X[:, 0]  # x axis
x2 = X[:, 1]  # y axis

plt.scatter(x1, x2, s=5)
# plt.show()

dbscan = DBSCAN(eps=0.25)
# eps = 0.5(default)
# metric : calculating method of distance(eps)
# min_sample : number of sample in a neighborhood for a point to be considered as a core point
# algorithm, n_jobs ...
dbscan.fit(X)

y_pred = dbscan.labels_.astype(np.int)
colors = np.array(['#ff0000', '#00ff00'])
plt.scatter(x1, x2, s=5, color=colors[y_pred])
# plt.show()

# result of KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.labels_.astype(np.int)

plt.scatter(x1, x2, s=5, color=colors[y_pred])
plt.show()






