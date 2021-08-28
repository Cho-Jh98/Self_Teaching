# PCA



### what is PCA

**PCA is acronom for Principle Component Analysis**

Principle componene : 주성분 벡터



PCA give us low dimensional representation of dataset

▶︎ Able to find linear combination of features or variables that are mutually uncorrelated
ex) Loan and LTI // Income and LTI >> has correlation >> delete feature

▶︎ Get rid of unnecessary feature while keep important one

▶︎ Can get good visualization + Can lower the dimension



#### HOW?

**Unsupervised learning approach**
No target or label. Just find pattern

1. Apply linear transformation to the data → minimize noise and redundanct
2. End up with principle component that contributes to describing pattern in the dataset
   (Some principles are more important than the other)

3. Covariance matrix can help this process



#### Covariance Matrix
* Diagonal item > have something to do with noise
* Off-diagonal item > have something to do with redundancy in the dataset

**PCA algorthm will produce set of principle components that**
▶︎ Maximize feature variance (→ reduce noise)
▶︎ Minimize covariance between pairs of feature(→ reduce redundancy)



#### PCA techniques

0. **First of all we have to normalize the data**

1. Calculate SVD(Singular Value Decomposition) of covariance matrix
2. Calculate eigenvectors



**Normalize**
: making sure features are normally distributed

mean = 0, variance = 1
It makes all feature value comparable

▶︎ Result: Principle components will be linearly independent and orthogonal



**Look at the sample distribution below**

<img src="https://user-images.githubusercontent.com/84625523/124494129-6f5aa380-ddf1-11eb-8476-a3a397603566.png" alt="PCA" style="zoom: 33%;" />

1. These features are postvily correlated
2. First principle component direction = eigenvector direction



​		**PCA in 3-D**

​		<img src="https://user-images.githubusercontent.com/84625523/124530925-63e69700-de48-11eb-8d11-d07154f1deb9.png" alt="PCA in 3-D" style="zoom: 150%;" />

#### PCA is Crucial in data visualization
: PCA reduces dimensionality of dataset(by using fewer features)

▶︎ Low dimensional representation of dataset
▶︎ The principle compenent(eigenvector) will represent the data

**Kaiser-criterion**
: How many principle components to keep?
**A)** We pick only the principle components that have eigenvalue greater than 1!!





### Example Code_digits

0. import library

   ```python
   from sklearn.decomposition import PCA
   from matplotlib import pyplot as plt
   from sklearn.datasets import load_digits
   ```



1. load dataset and label them

   ```python
   digits = load_digits()
   
   X_digits = digits.data
   y_digits = digits.target
   ```



2. fit the data to PCA estimator

   ```python
   estimator = PCA(n_components=2)
   
   X_pca = estimator.fit_transform(X_digits)
   ```

   We are going to construct covariance matrix and calculate eigenvector

   By using 2 n_compoents parameter we are able to visualize samples in 2 axis

   And also by using 2 components, if we print out X_pca.shape, result is

   > (1797, 2)



3. Assign color to samples

   ```python
   colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
   
   for i in range(len(colors)):
       px = X_pca[:, 0][y_digits == i]
       py = X_pca[:, 1][y_digits == i]
       plt.scatter(px, py, c=colors[i])  # assign given colors
       plt.legend(digits.target_names)
   ```

   0~9 respectively black ~ gray

   Scatter function will plot diffrent digit
   If digit is 'i' >> asign given color based on given index to the given sub data-set



4. Lable the axis and show the plot

   ```python
   plt.xlabel('first Principle Component')
   plt.ylabel('second Principle Component')
   
   plt.show()
   ```



5. Result

   ![plt.show()](https://user-images.githubusercontent.com/84625523/124497375-b9de1f00-ddf5-11eb-9350-a243b09c12c9.png)

* classifying '0' is not that hard to classify, also '3' and '6'
* But for example '1' and '7' is not that easy to separate
* Also '2' and '5'



6. Bonus. Explained variance

   ```python
   print("Explained variance: %s" % estimator.explained_variance_ratio_)
   ```

   ▶︎ Explained variance shows how much information can be attributed to the principle components.

   ​     **Result**

   > Explained variance: [0.14890594 0.13618771]

   1st principle component : ~15% variance

   2nd principle component : ~14% variance

   sum : ~29% of information is used

   ▶︎ It is not that bad result considering we minimized feature number from 64 to 2

   ▶︎ It is recomend to have more than 95% information(sum of principle component variance)

   ​	otherwise it doesn't have meaning that we are classifying samples.





## Example Code_mnist_784



0. library import

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import fetch_openml
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA
   ```



1. Load data-set

   ```python
   mnist_data = fetch_openml('mnist_784')
   
   features = mnist_data.data
   targets = mnist_data.target
   
   print(features.shape)
   ```

   > **Result**
   >
   > ```python
   > (70000, 784)
   > ```

   mnist_784 has 70,000 images with 28 x 28 pixel.



1. split the data-set into training data-set and target data-set

   ```python
   train_img, test_img, train_lbl, test_lbl = train_test_split(features, targets, test_size=0.15, random_state=123)
   
   scaler = StandardScaler()
   scaler.fit(train_img)
   train_img_before = scaler.transform(train_img)
   test_img_before = scaler.transform(test_img)
   ```

   scaler.fit() : calculate the mean and std for later scaling

   scaler.transfrom() : perform standardization by centering and scaling



2. Perform the analysis with PCA()

   ```python
   pca = PCA(.95)
   pca.fit(train_img)
   
   # Transform
   train_img_after = pca.transform(train_img_before)
   test_img_after = pca.transform(test_img_before)
   ```

   PCA(.95) : We keep 95% of variance → so 95% of the original information is used



3. Check the performance of PCA

   ```python
   print(train_img_before.shape)
   print(train_img_after.shape)
   ```

   >**Result**
   >
   >```python
   >(70000, 784)  # Original Data
   >(70000, 328)  # After PCA
   >```

   Approximately 400 features are removed while keeping 95% of original information.

   So instead of fitting 784 feature, we can use 328 feature(almost half of feature)



​		But... DL can use original feature as whole >> much powerful



#### Spolier

Algorithm Boosting!!!
