## DBSCAN

Density-Based Spatial Clustering of Application with Noise(DBSCN)

* Data Clustering algorithms such as K-means
* Density based → given a set of point in some place, it groups together that are closely packed together
* Very common clustering algorithm
* Out-perform K-means clustering



### DBSCAN Algorithm

* There are given points in 2-D space

* Trying to find every point → separated by distance no more than a given ε (Threshold distance)

* Same clusters: hop from a given node to another by hopping no more than ε.

  ⟹ Than the points are in the same cluster



### Illustration

![Illustration of K-means](https://user-images.githubusercontent.com/84625523/124546724-095c3380-de66-11eb-97ea-8dd59326b3a4.png)

### Advantages

* Can find non-linearly separable cluster(arbitarily shaped cluster)
* Unlike K-Means, we do not have to specify the number of cluster.
* Very robust to outliers
* Result does not depend on the starting condition
* Parameters are : ε + minimum number of neighbors
* O(N logN) of time complexity



### Disadvantage

* Not entierly deterministic
* Border points that are reachable from more than one cluster can be parts of either cluster
  * It depands on the order of the process
* Relies heavily on a distance measure of Euclidean-measure.
  * In higher dimension it is very hard to find a good value for ε
  * **Curse of Dimensionality**
* If the data and scale are not well undertood
  → choosing a meaningful ε can be difficult



## Example Code using make_moon data-set

0. Import library

   ```python
   import matplotlib.pyplot as plt
   from sklearn.cluster import DBSCAN, KMeans
   from sklearn import datasets
   import numpy as np
   ```

   We are going to use make_moon data-set.



1. Load data-set

   ```python
   X, y = datasets.make_moons(n_samples=1500, noise=.05)
   ```

   **parameters**

   * makes 2 interleaving circles.
   * n_sample
   * shuffle or not
   * noise
   * ect



2. Set x, y axis for plot and scatter the samples

   ```python
   x1 = X[:, 0]  # x axis
   x2 = X[:, 1]  # y axis
   
   plt.scatter(x1, x2, s=5)
   plt.show()
   ```

   > **Result**
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124550933-6529bb00-de6c-11eb-8c97-90b987b2cc71.png" alt="make_moon" style="zoom:36%;" />

   ​	As we can see it's easy to cluster those samples in to 2 groups, but for computer, it is not easy.

   ​	We are going to use DBSCAN method and K-Means method to compare it's performance

   

3. Set DBSCAN model and fit the samples

   ```python
   dbscan = DBSCAN(eps=0.25)
   dbscan.fit(X)
   ```

   * eps = ε : 0.25(default is 0.5)
   * metric = method of calculating ε
   * min_sample = number of smaples in a neighborhood for a point to be considered as core point
   * algorithm, n_jobs ect..



4. Predict the cluster and plot with color

   ```python
   y_pred = dbscan.labels_.astype(np.int)
   colors = np.array(['#ff0000', '#00ff00'])
   plt.scatter(x1, x2, s=5, color=colors[y_pred])
   plt.show()
   ```

   > **Result**
   >
   > ![DBSCAN](https://user-images.githubusercontent.com/84625523/124551363-f5680000-de6c-11eb-8d2b-cfb173e031db.png)



5. Result of K-Means

   ```python
   kmeans = KMeans(n_clusters=2)
   kmeans.fit(X)
   y_pred = kmeans.labels_.astype(np.int)
   
   plt.scatter(x1, x2, s=5, color=colors[y_pred])
   plt.show()
   ```

   > **Result**
   >
   > ![K-Means](https://user-images.githubusercontent.com/84625523/124551586-54c61000-de6d-11eb-8bed-61f004c85c0e.png)

   #### compare DBSCAN and K-Means

   we can see that DBSCAN has better performance than K-means method.

   But DBSCAN has some inconvenience that we have to set parameter ε.

   But still it's fair to say DBSCAN is great clustering method.
