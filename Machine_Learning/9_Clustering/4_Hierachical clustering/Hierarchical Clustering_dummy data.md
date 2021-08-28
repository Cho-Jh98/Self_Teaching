# Hierachical clsutering

## Why Hierachical clustering?

* We have huge disadvantage of k-means clustering : specifying **k** parameter in advance
* Hierachical clustering does not have that process
  * build a tree like structure out of the data which **contains** all the **k** parameter



### Hierachical clustering

* Method of cluster analysis which seeks to build hierarchy of cluster
* Agglomerative approach
  * bottom up
  * eache observations starts in its own cluster, and pairs of clusters are merged as one moves up the hierachy
* Merges and splits are determined in greedy manner
* Result is shown in dendrogram



### Algorithm

1. Start each node in its own cluster
   * sort of an initialization phase
2. find the two closes cluster and merge them together
3. repeat algorithm until all the points are in the same cluster
   * so there is only a single cluster left

**Q)** How can we measure the distance of two clusters?
**A)** Usually calculate the distance of the avarage of cluster's elements



### Illustration

<img src="https://user-images.githubusercontent.com/84625523/124553210-6e685700-de6f-11eb-872b-6541a4507689.png" style="zoom:33%;" />

* Euclidean-distance: Two observations are similar if the calculated distance is small(usual case)
* Correlation based distance: Two observations are similar if their features are highly correlated

<img src="https://user-images.githubusercontent.com/84625523/124554867-64475800-de71-11eb-91e3-a76addd20830.png" alt="Clustering" style="zoom:50%;" />

#### Dendrogram contains all the cases for k-means clustering!!

<img src="https://user-images.githubusercontent.com/84625523/124555267-de77dc80-de71-11eb-8ac8-9c9880b671c7.png" alt="K-means Clustering K value" style="zoom:50%;" />



#### Scaling of variable

* Important : scaling of varable matters
* We should use some way of standardization
  * Variables should be centered to have mean 0 or scaled to have std 1





## Example Code_dummy dataset

0. library import

   ```python
   import numpy as np
   from scipy.cluster.hierarchy import linkage, dendrogram
   import matplotlib.pyplot as plt
   ```



1. import dummy data-set and scatter

   ```python
   x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])
   
   plt.scatter(x[:, 0], x[:, 1], s=50)
   plt.show()
   ```

   > **Result**
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124557264-43cccd00-de74-11eb-9040-126c6a17530c.png" style="zoom:36%;" />



2. Calculate Linkage matrix and print

   ```python
   linkage_matrix = linkage(x, 'single')
   
   print(linkage_matrix)
   ```

   * Optimized algorithm based on minimum spanning tree > O(N^2)
   * Methods : single = nearest point algorithm
   * It is slow method. so unless we need dendrogram, use K-means or DBSCAN

   * all the information for dendrogram is in Linkage matrix

   > **Result**
   >
   > ```python
   > [[0.         1.         0.5        2.        ]
   >  [2.         4.         0.5        2.        ]
   >  [3.         5.         0.5        2.        ]
   >  [7.         8.         0.70710678 4.        ]
   >  [6.         9.         2.5        6.        ]]
   > ```
   >
   > So from dummy data-set
   >
   > [1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4] is respectively labeled 0 to 5.
   >
   > 1st line : [1, 1] and [1.5, 1] is merged.
   >
   > 2nd line : [3, 3] and [3, 3.5] is merged.
   >
   > 3rd line : [4, 4] and [3.5, 4] is merged.
   >
   > 7-th and 6-th is new sample which is merged beforehand.
   >
   > So the 7-th sample is a node that [3, 3] and [3, 3.5] is merged
   >
   > and 6-th sample is a node that [1, 1] and [1.5, 1] is merged



3. Construct Dendrogram

   ```python
   dendrogram = dendrogram(linkage_matrix, truncate_mode='none')
   plt.title("HC")
   plt.show()
   ```

   > **Result**
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124558080-2f3d0480-de75-11eb-83aa-5b1605f89553.png" alt="Dendrogram" style="zoom:36%;" />
