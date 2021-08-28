# Decision Tree



### Decision Tree란?

supervised learning으로 classification과 regression 모두 사용할 수 있다.

data/population을 질문을 통해 2개 이상의 집합으로 나누어 분류작업을 수행한다. 마지막 답이 남을 때까지 질문을 하고 분류한다. 마지막 남은 노드를 leaf node라고 하고 각 node들은 edge로 이어져 있다. 또한 가장 먼저 수행한 질문을 root node라고 한다.

![Decision Tree](https://user-images.githubusercontent.com/84625523/124384247-4e1f8780-dd0b-11eb-9dd3-38b54ae8d101.png)



이 과정에서 가장 중요한 것은 "어떤 질문을 해야 하나"이다. 이 질문에 대한 답을 찾는 방법은 아래와 같다.

* Gini index Approach
* Information entropy(ID3 algorithm or C4.5 apporach)
* algorithm based on variance reduction



#### 1. Information entropy(ID3 algorithm)

ID3 algorithm은 top-down greedy search 방식으로 decision tree를 그리는 방법 중 하나이다. 

Entropy를 계산해 정보가 얼마나 무질서하게 존재하는지를 숫자로 나타낸 것으로 이해했다. 따라서 이 entropy가 클수록 질문을 통해 sample들을 나눠야 하는 것이다. 계속 말하는 entropy는 Shannon-entropy로 확률변수 X가 x_1, x_2, ... x_n까지 존재하고 확률밀도함수 P(X)가 존재할 때 아래와 같은 식으로 계산된다.

![Shannon-entropy](https://user-images.githubusercontent.com/84625523/124384428-1ebd4a80-dd0c-11eb-89ad-82216391ea49.gif)

이 data-set의 성질에 따라 H(X)의 계산 결과가 결정된다.

* H(X) = 1 : 데이터들이 동일한 비율로 나뉜다.
  * ex) 참 거짓으로 분류하는 질문이 있을 때 절반은 참, 나머지 절반은 거짓이 된다.
* H(X) = 0 : 데이터들이 동일한 값을 갖고 있다.
  * ex) 참 거짓으로 분류하는 질문이 있을 때 모든 data들이 참, 혹은 거짓의 값을 갖는다.



위 계산과정을 거치고, 가장 큰 H(X)를 갖는 질문을 이용해 sample들을 split한다.



#### Gini Index

하지만 sklearn.tree 라이브러리에서 제공하는 DecisionTreeClassifier에서는 기본적으로 gini index를 사용한다. 이는 ID3 algorithm보다 계산 속도가 빠르기 때문이다. Gini index를 계산하는 식은 아래와 같다.

![Gini index](https://user-images.githubusercontent.com/84625523/124384931-86749500-dd0e-11eb-9b97-e7011e6d1b1b.gif)

다만 이 식의 값은 information entropy와 다르게 값이 가장 작은 feature로 나누게 된다.





### 장단점

모든 ML방법과 마찬가지로 장단점이 존재한다(ㅡㅡ)

#### 장점

* 직관적으로 질문을 통해 나누는 것이다 보니 이해하기가 편하다.

* 변수간 관계를 이용해 나누기에 적합하다.

* data preprocessing이 필요없다.

  ​	outlier에 의한 영향이 적다

* numerical, categorical variable에 적용이 가능하다.

  ​	Categorical variable이라 함은 Yes/No 혹은 Sunny/Cloudy/Rainy 등등으로 나뉘는 변수...



#### 단점

* Overfit하는 경향을 갖고 있다.

  ​	prunning, bagging 등의 방법을 통해 어느정도 해결이 가능하긴 하다.

* 작은 변화에 큰 영향을 받을 수 있다.

  ​	최적화된 tree를 구하기 위해 휴리스틱 해결법이 사용된다.





## 실습코드

실습코드는 매우매우매우매우 간단하다.



1. 라이브러리 import

   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.model_selection import cross_validate
   from sklearn.metrics import accuracy_score, confusion_matrix
   from sklearn import datasets
   ```

   DecisionTree를 이용할 것이기에 sklearn.tree에서 DecisionTreeClassifier를 가져오고,
   sklearn에서 제공하는 유방암 데이터를 이용할 것이다.



2. 데이터 불러오기, feature, label 저장

   ```python
   cancer_data = datasets.load_breast_cancer()
   
   features = cancer_data.data
   labels = cancer_data.target
   ```

   feature은 569 x 30 크기의 array
   label은 567 x 1 크기의 array이다.



3. train, test set 분리, 모델 생성

   ```python
   feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.2)
   
   model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
   ```

   criterion은 entropy, gini 두가지 방법으로 이용이 가능하고, max_depth는 decision node의 개수이다.(root node 포함)



4. prediction

   ```python
   predicted = cross_validate(model, features, labels, cv=10)
   print(np.mean(predicted['test_score']))
   ```

   test_score은 cv별 accuaracy를 반환한다. np.mean을 사용했음으로 10개의 accuracy의 평균값이 반환된다.



실행결과 : 0.931422...
