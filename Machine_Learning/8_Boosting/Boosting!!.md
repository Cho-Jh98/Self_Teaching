# Boosting



## Boosting?

부스팅이란 여러개의 weak learner을 결합해 예측, 혹은 분류 성능을 톺이는 알고리즘이다.

* classification과 regression 모두 적용이 가능하다.
* variance 와 bias를 줄일 수 있다.
* 크기가 작은 tree를 여러개 만든다.
  * 이렇게 만든 Tree들은 모두 독립적인 관계이다.
  * 부스팅은 sequential learning 알고리즘으로 이전단계의 정보를 사용한다
* 이는 추가적으로 정보를 찾아보다가 알게된 내용인데, 부스팅은 Additive model 중 하나라고 한다.
  * Additive model은 비모수 회귀, 즉 함수형태를 가정하지 않는 회귀모형을 의미한다.
  * 이러함 Additive model에 속한다는 것은 결국 회귀식 형태로 해석이 가능하다고 한다.



### Boosting is counter-intuitive theory

Weak learner is not able to make good prediction. (of course)

* weak learner is little bit better than random guess or toin toss. 50:50.
  * ex) decsion tree with depth 1
  * weak learner는 직역하자면 약한 학습기이다.
* Combine those weak learners, and it will be strong classifier
* By fitting small trees (decision stumps) we slowly improve the final result.
  * If it does not perform well, we will consider using "**AdaBoost**" algorithm



### Application

* Viola-Jones Face Detection
  * combines decision stumps to detect faces
  * weak learner decide whether the given section of the image contains a face or not
  * extremely accurate and fast



### Illustration

<img src="https://user-images.githubusercontent.com/84625523/124786830-0145e600-df83-11eb-8031-5ac1bd469189.png" alt="Illustration" style="zoom:50%;" />

* We want to classify those dots.

  * 2 x features + 2 output classes (yellow and green)

  1. Combine very simple weak learners such as depth 1 decision tree.
     * classifier made 2 mis-classification : 2 yellow dots are mis-classified
     * Boosting algorithm : in the next iteration, it will focus on the mis-classified items
  2. Focus ~ weight.
     * Increase **w**(weight) : for mis-classified items
     * Decrease **w** : for correctly classified items



### Boosting Algorithm formula_AdaBoost

* Strong Classifier which is combination of weak learners is H(x) model.

  ![](https://user-images.githubusercontent.com/84625523/124787873-d14b1280-df83-11eb-9efd-35674a1bb4d3.png)

  * Keep combine weak learner h(x)'s
  * assign +1 or -1 for the output classes(yellow and green)



* Initialize weight parameter at the beginning.

  ![weight initialize](https://user-images.githubusercontent.com/84625523/124788210-1e2ee900-df84-11eb-97a2-cb938c4c5220.png)

  * sum of all the weight is equall to 1.

  * and error is sum of misclassified weight.

    ![error](https://user-images.githubusercontent.com/84625523/124788552-6948fc00-df84-11eb-8085-b70fdd56cec0.png)

* Boosting algorithm works..

  1. Initialize weight.                  							  ![weight initialize](https://user-images.githubusercontent.com/84625523/124788210-1e2ee900-df84-11eb-97a2-cb938c4c5220.png)

     

  2. pick **h_t(x)** that minimize the error

  3. Calculate **ɑ_t**                                                    ![](https://user-images.githubusercontent.com/84625523/124790390-153f1700-df86-11eb-91aa-9ce3c707a29a.png)

  

  4. update **w_t+1**         									     ![](https://user-images.githubusercontent.com/84625523/124790667-50414a80-df86-11eb-923a-9bd27319cc67.png)
  5. On every iteration, we add new **h(x)** to the final model



#### ɑ value    ![](https://user-images.githubusercontent.com/84625523/124790390-153f1700-df86-11eb-91aa-9ce3c707a29a.png)

* Given h(x) classifier, **ɑ** vlaue increases as the error converges to 0
  * of course, good classifier are given more weight
* if **ɑ** value is 0 if error is 0.5
  * Because, if error is 0.5, it's a random guess.
  * We do not want our algorithm to rely on random guess
* Give negative **ɑ** for **h(x)** classifier that are wors than random
  * Which means it has higher error than random guess
* This **ɑ** parameter has something to do with **h(x)** learners!!



#### weight   ![](https://user-images.githubusercontent.com/84625523/124790667-50414a80-df86-11eb-923a-9bd27319cc67.png)

* Something to do with data-set
* Basic concept is setting higher weight for more important sample, and lower weight for less important ones
* **Z** make sure **w** is a distribution with sum of 1.
* **y(x)** function filps the sign of the exponent if **h(x)** is wrong
  * It makes sure to assign smaller weights to samples that are correctly classified 
    and bigger weights for mis-classified items
  * So at next iteration, next **h(x)** learner can focus on samples that are mis-classified
* **ɑ** parameter make sure that stronger classifiers' decision are more important.
  * Low error makes **ɑ** bigger, hence **w** greater.
  * If a weak classifier mis-classifies an input, we won't take it seriously (low **w**)



#### Features and Advantage

* The samples are weighted 
  * Some of them will occur more often
* Builds learners in sequential way (not independent)
* Final decision is the weighted average of **N** learners
  * better classifiers have higher weight of course..
* Reduces bias but increases over-fiting a bit



## Other Boosting Algorithm

* Gradient Boosting(GBM)
  * Sequential한 weak learner들을 residual을 줄이는 방향으로 결합하여 loss를 줄이는 아이디어
* XGBoost
  * GBM은 해당 train datad에 residual을 계속 줄이기 때문에 overfitting되기 쉽다.
  * 이를 해결하기 위한 XGBOOST는 GBM에 regularization term을 추가했다.
  * 또한 다양한 loss function을 지원해 task에 따른 유연한 튜닝이 가능하다.
* Light GBM
  * 부스팅 계열의 computational cost는 각 단계에서 weak learner인 best tree를 찾는데 쓰인다.
  * 예를 들어 100만개의 data를 iteration=1000으로 학습시킨 경우, 각 tree를 fit하기 위해 백만개의 데이터를 전부 scan 해야한다.
  * 이 과정을 1000 iteration으로 반복하니 computational cost와 시간이 너무 오래 걸린다.
  * Light GBM은 높은 cost 문제를 여러 알고리즘(histogram-based/GOSS/EFB)을 통해 tree 구축을 위한 scan 데이터 양을 줄인다.



## Example Code_Iris data

0. library import

   ```python
   from sklearn.ensemble import AdaBoostClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import confusion_matrix, accuracy_score
   from sklearn import datasets
   ```



1. Load Data

   ```python
   iris_data = datasets.load_iris()
   
   features = iris_data.data
   targets = iris_data.target
   ```



2. Split into training data-set, and test data-set & fit the model

   ```python
   feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)
   
   model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)
   model.fitted = model.fit(feature_train, target_train)
   ```

   * estimator : number of used weak learner
   * learning_rate : trade-off with estimator
   * random_state : random seed for base estimator
   * algorithm : SAMME, SAMME.R >> 1 or 2 additional assumption to make converge faster



3. Fit the prediction and print the performance

   ```python
   model.prediction = model.fitted.predict(feature_test)
   
   print(confusion_matrix(target_test, model.prediction))
   print(accuracy_score(target_test, model.prediction))
   ```

   > Result
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124793543-0f970080-df89-11eb-8ebe-508169b7f364.png" alt="Result" style="zoom:80%;" />
   >
   > we can see that it acheived 100% accuracy





## Example Code_wine.csv



0. Library import

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.ensemble import AdaBoostClassifier
   from sklearn.model_selection import GridSearchCV
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import confusion_matrix, accuracy_score
   from sklearn import preprocessing
   ```



1. Define output parameter

   ```python
   def is_tasty(quality):
       if quality >= 8:
           return 1
       else:
           return 0
   ```

   wine data has several features for tasteness of wine and has score for quality.

   We are going to say if quality is equall or higher than 8, it is tasty wine.



2. Load Data & set features for prediction

   ```python
   data = pd.read_csv("wine.csv", sep=";")
   
   features = data[
           ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
   data['tasty'] = data["quality"].apply(is_tasty)
   
   targets = data['tasty']
   print(features.shape)
   print(data['tasty'].shape)
   ```

   > Result
   >
   > ```python
   > (4898, 11)
   > (4898,)
   > ```
   >
   > It shows that we have 11 features with 4898 samples.
   >
   > data['tasty'] has 1-dimensional array with 1 and 0 (tasty or not)



3. Reshape for machine learning to handle arrays

   ```python
   X = np.array(features).reshape(-1, 11)
   y = np.array(targets).reshape(-1, 1)
   ```



4. Preprocess with MinMaxScaler() and split the data into training data-set and test data-set

   ```pyhton
   X = preprocessing.MinMaxScaler().fit_transform(X)
   
   feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)
   ```



5. Set the parameter grid and search.

   ```python
   param_dist = {
       'n_estimators': [10, 50, 200],
       'learning_rate': [0.01, 0.1, 1]
   }
   
   estimator = AdaBoostClassifier()
   
   grid_search = GridSearchCV(estimator=estimator, param_grid=param_dist, cv=10)
   grid_search.fit(feature_train, target_train)
   ```

   * Use estimator AdaBoostClassifier.
   * We are gonna search 2 parametr. n_estimator and learning_rate.



6. Test the prediction & print performance

   ```python
   prediction = grid_search.predict(feature_test)
   
   print(confusion_matrix(target_test, prediction))
   print(accuracy_score(target_test, prediction))
   ```

   > Result
   >
   > <img src="https://user-images.githubusercontent.com/84625523/124795037-9c8e8980-df8a-11eb-9f37-eb4b270c4e08.png" style="zoom:50%;" />
   >
   > We can see the result that the accuracy score is about 96%























