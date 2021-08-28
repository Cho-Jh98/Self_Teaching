# Support Vector Machine(SVM)

* Support Vector Machine: Support vector와 hyper-plane이 주요 개념인 머신러닝 알고리즘

* Classification, Regression에 사용 가능

  * 특히 pattern recognition에 유용하다.

    Cancer, genetic Diseases identifying
    Text classification
    Detecting rare events

* Supervised Learning

## Linear SVM

###  Hyper-plane

쉽게 말하면 집단을 나누는(분류) 직선(평면)이다. 하지만 이 평면에는 특징이 있다. 분류할 때 최고의 마진을 가져가는 방향으로 진행된다. 마진이 크면 클수록 학습에 사용하지 않은 데이터가 들어오더라도(test-set) 잘 분류할 가능성이 커진다.

아래 그림에서 빨간 점과 파란 점을 최대한의 마진으로 나누는 직선은 검은색 직선이다. 그림의 예시는 2차원이지만 만약 3차원 이상이라면 평면이 되고, 이를 hyper plane이라고 한다.

<img src="https://user-images.githubusercontent.com/84625523/124222765-575af980-db3d-11eb-8fe3-c29d8c883e59.png" alt="SVM_1" style="zoom:50%;" />

직선과 가장 가까운 점들을 support vector라고 하고 마진의 폭(width, margin)은 1/||w||이며 이를 최대로 하기 위한 과정은 ||w||를 최소화하는 과정과 같고, 이 과정이 바로 SVM optimization이다.



### Slack 변수

sample들이 잘 나뉘어져 있으면 좋겠지만 만약 겹치는 부분이 생기면 어떨까? 다음 그림을 보자.

![SVM_2](https://user-images.githubusercontent.com/84625523/124223443-b2d9b700-db3e-11eb-9021-56bad0277f84.png)

파란 점들과 빨간 점들이 어느정도 분리되어 있지만 겹치는 부분 또한 존재한다. 이런 상황에서 구분하는 선을 그을 때 객관적이지 못할 수 있다. 이를 피하기 위해 slack 변수를 사용한다. slack 변수는 𝜉𝑖를 사용하고, 이를 최소화 하는 방향으로 hyper-plane을 구한다.

<img src="https://user-images.githubusercontent.com/84625523/124223701-33001c80-db3f-11eb-9f3a-96645871c6ff.png" alt="SVM_3" style="zoom:33%;" />

이를 고려한 SVM 최적화는 아래 식으로 표현된다.

![SVM_4](https://user-images.githubusercontent.com/84625523/124224313-5d060e80-db40-11eb-8b2b-86459b44a12c.gif)

이 때 C를 regularization parameter이라고 하는데, 이는 overfitting되지 않기 위해 들어가는 일종의 penalty항이다.

C값은 조절이 가능하다.

- C가 큰 경우 : 알고리즘은 100%를 분류하기 위해 노력할 것이다.(overfitting)
- C가 작은 경우 : margin의 폭이 커져서 잘못 분류된 sample이 생길 수 있다.





## Non linear SVM

지금까지는 선형적으로 분류가 가능한 경우만 보았다. 하지만 선형적으로 분리가 되지 않는 경우는 어떨까?

<img src="https://user-images.githubusercontent.com/84625523/124224833-59bf5280-db41-11eb-9175-1202360351fc.png" style="zoom:50%;" />

이러한 경우 구별이 가능한 방향으로 mapping을 시키면 새로운 공간에서 구별이 가능하게 된다. 이해가 쉽게 차원을 늘린다고 생각하면 된다. 

<img src="https://user-images.githubusercontent.com/84625523/124224961-98eda380-db41-11eb-8b03-37a58ece6891.png" style="zoom: 50%;" />

이렇게 mapping을 시킬 때 사용하는 함수들을 Kernel function이라 하고, 이 과정을 Kernel trick이라고 한다.



### Kernel Function(Kernel trick)

앞서 말했듯이 낮은 차원에서 분류가 불가능한 sample들을 mapping을 통해 높은 차원으로 변환시킨 뒤, 선형적으로 구별이 가능하는 방법을 Kernel trick이라고 한다. 이 때 사용하는 함수를 Kernel function이라고 한다.

많이 사용되는 함수는 아래와 같다.

![](https://user-images.githubusercontent.com/84625523/124225362-4b256b00-db42-11eb-9d84-50c2ef310310.png)



## Mathmatics in SVM

![](https://user-images.githubusercontent.com/84625523/124225714-e0286400-db42-11eb-9e23-976a7a8034cf.png)