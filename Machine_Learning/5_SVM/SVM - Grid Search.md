# Grid Search for SVM

SVM에 있는 대표 hyperparameter은 두가지가 있다. 각각 C와 gamma인데, 그 특징에 대해 알아보자



## C parameter

#### C parameter은 margin과 training error에 대한 trade-off를 결정한다.



우리가 찾는 hyper-plane에서 cost-function은 다음과 같이 나타낼 수 있다.

![SVM_4](https://user-images.githubusercontent.com/84625523/124224313-5d060e80-db40-11eb-8b2b-86459b44a12c.gif)

여기서 

![](https://user-images.githubusercontent.com/84625523/124376597-bf971000-dce2-11eb-9b92-5312f6014887.png)

는 slack variable이라고 불리며 벗어난 만큼을 추가해 trainig error를 허용한다.
따라서 C는 margin과 training error에 대한 trade-off를 결정해주는 tuning parameter인 것이다.

* C⬆︎ : training error을 조금만 허용하지 않는다. ⇒ Overfitting
* C⬇︎ : training error을 많이 허용한다. ⇒ Underfitting



## Gamma parameter

#### Gamma parameter는 Radial Bias Kernel을 선택했을 때 tuning parameter이다.



kernel의 종류는 크게 linear, polynomial, (gausian)radial bias function(rbf) kernel등이 있다. 그 중 rbf를 사용했을 때 사용되는 hyperparameter가 gamma parameter이다.

Raidal bias kernel은 아래와 같은 수식을 갖는다.

![Raidal Bias Kernel](https://user-images.githubusercontent.com/84625523/124376936-44cef480-dce4-11eb-98cd-bb96634964b4.png)



그래프를 확인하기 위해 1차원이라고 생각하고 z = x - y 라고 두고 그래프를 그리면 아래와 같다.

![rbf](https://user-images.githubusercontent.com/84625523/124377054-fa01ac80-dce4-11eb-89f7-0ea02b76e513.png)

위 그래프에서 볼 수 있듯이 𝛄는 하나의 데이터가 영향력을 행사하는 거리를 결정해준다.

* 𝛄⬆︎ : 각 데이터의 영향력이 행사하는 거리 감소 ⇒ Overfitting
* 𝛄⬇︎ : 각 데이터의 영향력이 행사하는 거리 증가 ⇒ Underfitting





## 코드 실습



0. 라이브러리 import

   ```python
   from sklearn import svm
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import confusion_matrix
   from sklearn.metrics import accuracy_score
   from sklearn import datasets
   from sklearn.model_selection import GridSearchCV
   ```

   sklearn에 있는 iris data를 이용할 예정이다.



1. data 불러오기 & feature, target 설정

   ```python
   iris_data = datasets.load_iris()
   
   features = iris_data.data
   target = iris_data.target
   ```



2. train_test_split & model 설정

   ```python
   feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)
   
   model = svm.SVC()
   ```

   train_test_split은 random하게 data를 나누고 그로 인해 실행할 때마다 다른 결과가 나온다.

   model은 SVC 모델을 사용한다.



3. prarameter grid 설정

   ```python
   param_grid = { 'C' : [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200], 
   							 'gamma' : [1, 0.1, 0.01, 0.001], 
   							 'kernel': ['rbf', 'poly', 'sigmoid']}
   
   ```

   kernel 함수에는 rbf, poly, sigmoid를 넣었고,
   C 와 gamma에는 각각 원하는 숫자를 넣으면 된다.

   이 경우 12 * 4 * 3 가지수의 가능한 조합이 있고, 이 144가지의 조합을 모두 이용해서 가장 높은 accuracy를 반환할 것이다.



4. model fitting with param_grid

   ```python
   grid = GridSearchCV(model, param_grid, refit=True)
   grid.fit(feature_train, target_train)
   ```



5. best estimator 반환 및 accuracy 출력

   ```python
   print(grid.best_estimator_)
   
   grid_prediction = grid.predict(feature_test)
   print(confusion_matrix(target_test, grid_prediction))
   print(accuracy_score(target_test, grid_prediction))
   ```

   출력결과 VV

   <img src="https://user-images.githubusercontent.com/84625523/124377666-0affed00-dce8-11eb-89ef-5b74c77956ac.png" alt="best_estimator_and_accuracy" style="zoom:50%;" />

   이 결과는 출력할 때마다 다르게 나올 것이다.





Iris dataset은 sample 수도 적고, 분류가 잘 되도록 가공되어있는 데이터이다. 그렇기에 높은 정확도 값이 나올 수 있다.