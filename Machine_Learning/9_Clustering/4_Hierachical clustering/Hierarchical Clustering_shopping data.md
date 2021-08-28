## Optimal k Clusters (Hierarchical Clustering)



### Use the Dendrogram

1. Determine the largest vertical distance(gap) that does not interact any of the other clusters
2. Draw a horizontal line at the top and at the bottom
   * It means at both extremities
3. Count the number of vertical lines going through the horizontal line
4. That is the optimal **k** number of cluster

#### Illustration

<img src="https://user-images.githubusercontent.com/84625523/124703413-3aec0200-df2d-11eb-8186-718e3872ebe9.png" alt="Find optimal k-value" style="zoom:33%;" />

 



### Example Code

0. Library import

   ```python
   import matplotlib.pyplot as plt
   import pandas as pd
   import scipy.cluster.hierarchy as cluster_algorithm
   from sklearn.cluster import AgglomerativeClustering
   ```



1. Data loading and allocate feature

   ```python
   shopping_data = pd.read_csv('shopping_data.csv')
   print(shopping_data)
   data = shopping_data.iloc[:, 3:5].values
   print(data.shape)
   ```

   clustering feature will be 'Annual Income' and 'Spending Score'

   > Result
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124704158-97035600-df2e-11eb-818d-f96b7771a898.png" alt="print" style="zoom: 33%;" />
   >
   > as we can see data[:, 3:5] will be 'Annual Income' and 'Spending Score'



2. Construct a dendrogram with linkage matrix

   ```python
   plt.figure(figsize=(10, 7))
   plt.title("Market  Segmentation Dendrogram")
   dendrogram = cluster_algorithm.dendrogram(cluster_algorithm.linkage(data, method='ward'))
   ```

   figsize is size of pop-up window that shows pyplot

   > Result
   >
   > ![Dendrogram](https://user-images.githubusercontent.com/84625523/124704456-12650780-df2f-11eb-8792-bc13c611cff5.png)
   >
   > If we want to find optimal k value, we should look for maximum gap between splitst
   >
   > It will be 3rd ~ 4th, counting from the top.
   >
   > We can optimize the k value with size of 5.



3. Adopt Hierachical clustering and plot.

   ```python
   cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
   cluster.fit_predict(data)
   
   plt.figure(figsize=(10, 7))
   plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
   plt.title('Market Segmentation')
   plt.xlabel('Annual income')
   plt.ylabel('Spending Score')
   
   plt.show()
   ```

   > Result
   >
   > ![Clustering](https://user-images.githubusercontent.com/84625523/124704735-81426080-df2f-11eb-84a9-f0ff136bd9ef.png)
