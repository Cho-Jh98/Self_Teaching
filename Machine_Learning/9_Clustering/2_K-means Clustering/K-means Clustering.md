## K-Means Clustering



### K-Means Clustering?

* Unsupervised learning algorithm in Data Mining.

* Automatically divide data into clusters and groupings of similar item + without being told what groups should look like.

* **Q)** How could computer know where one group starts and the other ends?
  **A)** Element inside a cluster should be similar to each other, but very different from those outside.
       Make decision based on similarity

* Partition of **n** observations into **k** clusters in which each observation belongs to cluster with the nearest mean.
* Can be done with..
  * graph algorithms construct the minimum spanning tree and remove the last k edges
  * NP-hard problem
  * Lloyd-algorithm



### Lloyd-algorithm

1. initialize the centroid at random

   * centroid : center of a given cluster

2. decide for every point in our dataset

   : Which centroid is the nearest to them

3. calculate the new means of every distinct cluster

   : 2~3 run until convergence







### with graph..

<img src="https://user-images.githubusercontent.com/84625523/124538900-64d2f500-de57-11eb-87a5-dbd5463920a8.png" style="zoom: 33%;" />

Nearest is the green one >> yellow sample is going to be green dot and we do it over and over again..

Until all the samples are labeled

<img src="https://user-images.githubusercontent.com/84625523/124539072-baa79d00-de57-11eb-98fc-3a9783c2d156.png" style="zoom:33%;" />



And then we update the centroid

<img src="https://user-images.githubusercontent.com/84625523/124539398-5fc27580-de58-11eb-9a7b-5f222cd17155.png" alt="centroid update" style="zoom:33%;" />

But look at the blue dot in the middle of red cluster and blue cluster. it is more close to red cluster. So we re-label it with red, and update a centroid of red cluster



<img src="https://user-images.githubusercontent.com/84625523/124539180-edea2c00-de57-11eb-8236-f8fdaa326806.png" style="zoom:33%;" />

This is final result of k-means clustering





### How to find **k** parameter?

* w/ priori knowledge: we know how many clustes we want to construct
* w/o priori knowledge: K ~ n/2
  * n is number of element in the dataset
* **Elbow method** : Monitor the change of homogeneity within the clusters with different **k** values
  * **Percentage of variance explained** represents preformance of number of clusters
    : one should choose a number of clusters.
      And that point, adding another cluster does not give much better modeling of data
  * The **Elbow Point**



#### Elbow point

<img src="https://user-images.githubusercontent.com/84625523/124540207-df9d0f80-de59-11eb-9a22-ab1ed5b348a8.png" alt="Elbow point" style="zoom:33%;" />

In this cas **k** = 4



### Pros and Cons

#### Pros

* Relies on simple principle to identify cluster
  * uses kNN
* Felxible
* Efficient

#### Cons

* Not so sophisticated
* Use an element of random chance → not guaranteed to find the optimal set of cluster
* **k** parameter : we have to know in advancne how many cluster we want to find



### Clustering and Classification

* Clustering is different from classification or numerical prediction
  * Classification / regression : result is from model with training features and targets
  * Clustering
    * creating new data
    * Unlabeld examples are given a cluster labeld which is inferred entirely from relationship of data





## Example Code_make_blobs datasets

0. Import library

   ```python
   import matplotlib.pyplot as plt
   from sklearn.cluster import KMeans
   from sklearn.datasets import make_blobs
   ```



1. load data-set

   ```python
   x, y = make_blobs(n_samples=1000, centers=10, random_state=0, cluster_std=3)
   ```

   n_sample : total number of point(samples)
   random_state : random number generation
   centers : number of cluster, fix center location
   x: coordinance, y: center



2. scatter the samples

   ```python
   plt.scatter(x[:, 0], x[:, 1], s=50)
   plt.show()
   ```

   > **Result**
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124544469-1d059b00-de62-11eb-83dc-b78dec9dad69.png" alt="scatter" style="zoom:33%;" />



3. modeling of KMeans

   ```
   model = KMeans(5)
   model.fit(x)
   y_kmeans = model.predict(x)
   ```

   KMeans(n) : n is the number of cluster



4. scatter the KMeans clustering

   ```python
   plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='rainbow')
   plt.show()
   ```

   > **Result**
   >
   > ![Scatter](https://user-images.githubusercontent.com/84625523/124544696-8b4a5d80-de62-11eb-8d81-6d70ad922902.png)

   we can see 5 colors as we set KMeans(5)
