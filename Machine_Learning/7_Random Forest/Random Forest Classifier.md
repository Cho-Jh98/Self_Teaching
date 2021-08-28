# Random Forest

앞서 Decision Tree의 단점에 대해서 알아봤었다. 가장 큰 단점은 over-fit하는 경향이 매우 크다는 점이다. ML 알고리즘의 목표는 training data-set에서의 경향성을 파악해 이를 test-set(unseen data)와 맞추는 것이다. 하지만 이 둘을 동시에 성취하기란 매우 힘들다. 이는 bias-variance trade-off라고 부른다.



#### Bias

: learning algorithm에서 잘못 분류한 에러

* Bias가 크면 feature와 target의 경향을 놓친다(underfitting)

* 모델 mismatch에 의한 에러.

#### Variance

: 작은 변화로 인해 생기는 에러

* Variance가 크면 작은 변화에도 예민하게 반응함으로 overfitting이 될 가능성이 높아진다.

#### Bias-Variance trade-off

Bias와 Variance 둘을 동시에 최적화하기 매우 어렵다.

기본적으로 bias를 줄이면 variance가 커지고, 반대로 variance를 줄이면 bias가 커진다.

<img src="https://user-images.githubusercontent.com/84625523/124412968-a21d8100-dd8a-11eb-841b-358eafb8c9f7.png" alt="Bias-Variance Trade-off" style="zoom:50%;" />

x축에서 왼쪽으로 갈 수록 under-fitting이 되고, 이러한 알고리즘으로는 linear regression이 있다.
반대로 오른쪽으로 갈 수록 over-fitting이 되고, 이러한 알고리즘으로는 decision tree가 있다.

Decision Tree에서 이러한 over-fitting을 막기 위해 Pruning, Bagging등의 방법을 사용한다.



## Pruning and Bagging



### Pruning

이해하기 쉽게 가지치기라고 생각하면 편하다. Tree는 기본적으로 불안정해서 data를 조금만 교란시켜도 결과가 크게 바뀐다. 다시 말하면 bias가 낮고, variance가 크다(data의 변화에 예민). 
Variance를 줄이기 위해 split을 줄이고 tree의 크기도 주려서 bias를 약간 높여서 더 좋은 예측 결과를 기대할 수 있다.

<img src="https://user-images.githubusercontent.com/84625523/124413799-566bd700-dd8c-11eb-9b52-cf040dd2661b.png" alt="Prunning" style="zoom:50%;" />



### Bagging

하지만 pruning은 Bias가 커지기 때문에 잘 사용되지는 않는다. Bagging은 pruning을 조금 발전시킨 방법이라고 이해했다. 작은 tree들을 모아서, 분류를 야금야금 함으로써 좋은 분류를 할 수 있는 것이다. 집단지성으로 이해하면 편할 것 같기도 하다.

Bagging은 Bootstrap Aggregation의 약자이다. Bootstrap은 통계학 기법중 하나로 sample data들을 복원추출로 일정 수를 뽑아서 우리가 원하는 통계량을 계산하는 행위를 N번 반복하여 평균과 분산을 계산해 이용하는 방법이다. Bagging도 마찬가지이다.

Data-set에서 sample들을 뽑아 Tree를 만들고 prediction을 평균을 내는 행위를 반복하는 것이다. 이 때 Tree들은 leaf node까지 다 자란 unpruned tree이다. Bagging을 이용하면 추가적인 bias 없이 variance를 줄일 수 있다.
Regression 에서는 평균을 내면 되고
Classification 에서는 가장 많이 뽑힌 수를 세면 된다.



## Random Forest

하지만 Bagging 역시 단점이 존재한다. 바로 Correlation이 생긴다는 것이다. 이게 무슨 뜻이냐, data-set에는 가장 중유한 분류 기준이 있을 것이고, 많은 Tree들이 그 기준에 따라 자란다는 것이다. 다시 말하면 tree의 모양이 비슷하게 자란다. 



Random Forest는 이러한 correlation을 없애고 bagging보다 더 작은 variance를 기대할 수 있다. 방법은 bagging과 비슷하지만, 가장 큰 다른 점은 split을 하는 기준을 무작위로 선택한다는 점이다. Feature/predictor을 random selection을 통해 뽑고 이 수는 전체 feature의 제곱근과 같다.

다시말하면 N개의 feature가 있을 때, bagging은 N개의 feature을 모두 이용하지만, Random Forest는 √N개만 사용한다.





## 예시 코드

1. 라이브러리 import

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_validate
   ```

   credit_data.csv에 있는 데이터를 이용해 default값을 예측할 것이다.



2. 데이터 불러오기 & feature, target 설정

   ```python
   credit_data = pd.read_csv("credit_data.csv")
   
   features = credit_data[["income", "age", "loan"]]
   targets = credit_data.default
   ```

   feature로는 income, age, loan을 이용할 것이다.



3. data-frame >> array

   ```python
   # machine learning handle arrays, not data-frames
   X = np.array(features).reshape(-1, 3)
   y = np.array(targets)
   ```



4. model fitting & prediction

   ```python
   model = RandomForestClassifier()
   predicted = cross_validate(model, X, y, cv=10)
   
   print(np.mean(predicted['test_score']))
   ```

   출력 결과 99%에 가까운 accuracy를 보인다. 지금까지 logistic regression은 93%, kNN은 97~8%의 정확도를 보였다.

   Random Forest의 성능을 보여주는 좋은 지표이다.







** bonus

#### Grid Search

0. 라이브러리 import

   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import GridSearchCV, train_test_split
   from sklearn.metrics import confusion_matrix, accuracy_score
   from sklearn import datasets
   ```

   sklearn의 digits dataset을 이용할 것이다.

   또한 sklearn.model_selection의 GridsearchCV를 이용해 최적의 parameter을 찾아보자.



1. 데이터 불러오기

   ```python
   digit_data = datasets.load_digits()
   
   image_features = digit_data.images.reshape((len(digit_data.images), -1))
   image_targets = digit_data.target
   ```

   datasets에서 digits을 불러오고 이를 feature와 target으로 각각 저장한다.



2. 모델 설정 & train, test set 설정

   ```python
   random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='auto')
   
   
   feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=0.2)
   ```

   n_jobs는 몇개의 코어를 사용할지 정하는 parameter. -1로 설정하면 모든 코어를 사용한다는 것이다. 계산량이 많기 때문에 모든 코어를 사용해주자. 2~3번 돌리면 노트북이 뜨끈뜨끈 해진다.

   max_feature은 몇개의 feature을 사용해 split할지를 정해주는 것이다. √N개를 사용하기 위해 auto로 값을 지정해준다.물론 sqrt로 해도 동일한 결과가 나온다.



3. paramater grid 설정

   ```python
   param_grid = {
       "n_estimators": [10, 100, 500, 1000],
       "max_depth": [1, 5, 10, 15],
       "min_samples_leaf": [1, 2, 4, 10, 15, 30, 50]
   }
   ```

   다른 모든 model과 동일한 문법이다. 원하는 parameter과 그에 해당하는 값들을 dictionary로 저장해준다.



4. Grid search

   ```python
   grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
   grid_search.fit(feature_train, target_train)
   ```

   GridSearchCV 함수를 이용해 parameter들을 찾아보자. 시간이 조금 걸린다. 잠시 일어나서 스트레칭도 좀 하고 그러자..



5. parameter print

   ```python
   print(grid_search.best_params_)
   
   optimal_estimator = grid_search.best_params_.get("n_estimators")
   optimal_depth = grid_search.best_params_.get("max_depth")
   optimal_leaf = grid_search.best_params_.get("min_samples_leaf")
   
   print(optimal_leaf, optimal_depth, optimal_estimator)
   ```

   최적화된(accuracy가 가장 높은) parameter을 출력해보자. 출력 결과는 맨 아래서 한꺼번에 보자.

   맨 위에 print문은 dictionary 형태로 출력이 되고, 각각의 parameter들을 변수를 이용해 저장할 수 있다.

   

6. Accuracy score, confusion matrix

   ```python
   grid_prediction = grid_search.predict(feature_test)
   
   print(confusion_matrix(target_test, grid_prediction))
   print(accuracy_score(target_test, grid_prediction))
   ```

   prediction을 정하고 정확도 관련 값들을 출력해보자



![Random Forest Grid Search](https://user-images.githubusercontent.com/84625523/124415776-7e5d3980-dd90-11eb-9c88-0edaff45a77d.png)

정확도가 꽤나 괜찮게 나온다. 





##### spoiler

bagging과 random forest 보다 더 좋은 방법이 있다.. 항상 더 좋은 방법이있지... 근데 이것들 보다 DL이 더 좋겠지...
