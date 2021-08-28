import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=1000, centers=10, random_state=0, cluster_std=3)
# n_sample : total number of point
# random_state : random number generation
# centers : number of center, fix center location
# x: 좌표, y: center

# print(x)


plt.scatter(x[:, 0], x[:, 1], s=50)
# plt.show()

model = KMeans(5)  # number of cluster inside ()
model.fit(x)
y_kmeans = model.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='rainbow')
plt.show()


