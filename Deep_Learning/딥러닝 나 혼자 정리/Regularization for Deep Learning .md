# Regularization for Deep Learning

ref) Ian Goodfellow, "Deep Learning", chap 7. Regularization for deep learning



### Purpose

* 머신러닝의 주요 문제는 "training data 뿐만 아니라 new input 또한 어떻게 잘 맞추는가" 이다.  새로운 데이터를 잘 맞추기 위해서는 over fitting을 해결해야하며 이 해결책 중 하나가 정규화이다. 다양한 정규화 방법 중 하나는 모델의 복잡도가 높아질수록 불이익(penalty)를 주는 것이다. 

* 정규화(regularization)은 training error을 줄이는게 아니라, generalization error을 감소시키는 것이라고 정의한다.

  > We deﬁned regularization as “any modiﬁcation we make to alearning algorithm that is intended to reduce its generalization error but not its training error.”

### 1. Parameter Norm Penalties

* 비용함수에 제곱을 더하거나(L2), 절댓값을 더해서(L1) 웨이트의 크기에 제한을 준다.

* Loss function에 weight가 커질 경우에 페널티 항목을 집어 넣는 방법.

  * L2 weight decay(제곱값)

    <img src="https://user-images.githubusercontent.com/84625523/126283011-e763912e-e25a-4de8-9ecc-0cd77c29478b.png" style="zoom:30%;" />

    * 오차의 제곱의 합
    * Least squares error(LSE)

    L2 parameter는 ridge 회귀분석 및 tikhonov 정규화로도 알려져 있다.

  * L1 weight decay(절댓값)

    <img src="https://user-images.githubusercontent.com/84625523/126282691-87a2046e-3a9f-45b3-86e6-0d57413b8b4e.png" style="zoom:30%;" />

    y_i : 실제 값

    f(x_i) : 예측치

    * 실제 값과 예측치 사이의 오차값의 절대값을 구하고 그 오차의 합을 L1 Loss라 한다.
    * Least absolute deviations(LAD), Least absolute Error(LAE), Least absolute value(LAV) 등등 여러가지로 불린다.

    LASSO는 선형모델의 L1 페널티와 최소제곱법(LSM)을 합친 모델이다.

* 결론

  * L1, L2 regularization은 모두 overfitting을 막기 위해 사용된다.
  * L1은 sparse model(coding)에 적합하다. 특히 convex optimization에 유용하게 쓰인다고 한다.
  * L1은 미분 불가능한 점이 있기 때문에 gradient base learning에는 주의가 필요하다.





### 2. Dataset Augmentation

* 머신러닝에서 가장 효과적인 정규화 방법은 training set의 크기를 늘리는 것이다.
  이와 관련해 training set에 가짜 데이터(face data)를 포함할 수 있다.
* Augmentation 방법
  * 이미지 반전
  * 이미지 밝기 조절
  * subsampling
  * noise 넣기
* 주의사항
  * 데이터를 변환할 때, 데이터의 특징을 고려해야 한다.
    ex) b, d 혹은 6, 9처럼 180도를 뒤집은 것과 같은 데이터의 경우 좌우 반전하여 데이터를 늘리는 것은 적절하지 않다.





### 3. Noise Robustness

* Robust란?
  * 머신러닝에서 generalization이란 일부 특정 데이터만 잘 설명하는 overfitting적인게 아니라 범용적인 데이터에 적합한 모델을 의미한다. 즉, 일반화를 잘 하기 위해서는 이상치나 노이즈가 들어와도 크게 흔들리지 않아야(=robust) 한다.

* 레이어 중간에 noise를 추가(noise injection)하는게 파라미터를 줄이는 것보다 강력할 수 있다.
  * weight에도 noise를 넣는다. 이는 hidden layer를 Dropout하는 것 보다 덜 엄격한 느낌이 든다.
  * classification을 할 때 label-smoothing을 한다.
    ex) (1, 0, 0) →(0.8, 0.1, 0.1)





### 4. Semi-Supervised Learning

* 비지도 학습 + 지도 학습
* 딥러닝에서 representation을 찾는 과정
  CNN에서 convolution과 subsampling이 반복되는 과정인 feature extraction이 일조으이 representation을 찾는 과정이다. Convolutional layer을 정의할 때 사전학습(pre-training)을 하면 비지도 학습에 적합한 representation을 한다. (AutoEncoder가 적합한 방법이다.)





### 5. Multi-Task Learning

* 한번에 여러 문제를 푸는 모델
  ex) 얼굴을 통해서 나이/성별 분류, 한번에 수학/과학 문제를 푸는 과정

* 같은 input이지만 목적에 따라 나눠지는 구조이다.

  <img src="https://user-images.githubusercontent.com/84625523/126288841-5ca3cac9-001b-4b80-bb7d-cbb8ee803fe7.png" style="zoom: 33%;" />

  * Shared 구조 덕분에 representation을 잘 찾아준다.
  * 서로 다른 task(문제) 속에서 몇 가지 공통된 중요한 factor(요인)이 뽑히고 shared 구조를 통해 representation을 찾을 수 있다. 또한 이 구조 덕분에 각각의 요인을 따로 학습시킬 때보다 더 좋은 성능을 낸다.
    최근 Google의 NLP에서는 감정분석/번역 등 다양한 Multi-Task Learning을 통해 모델 성능이 좋아진다는 연구를 발표하기도 했다.

* 딥러닝 관점에서 Multi-Task Learning을 하기 위해선 모델의 training set으로 사용되는 변수는 연관된 다른 모델의 변수와 두 개 이상 공유한다는 가정이 존재한다.

  > From the point of view of deep learning, the underlying prior belief is the following: among the factors that explain the variations observed in the data associated with the different tasks, some are shared across two or more tasks.



### 6. Early Stopping

* 말 그대로 일찍 종료한다는 것이다.
* 학습셋의 오류는 줄지만 검증셋 오류가 올라가면 멈추는데, 이는 overfitting을 방지하는 방법 중 하나로 간단하고 효과적이어서 정규화 방법으로 많이 활용된다.
  * 이전 epoch와 비교해서 오차가 증가하면 overfitting이 발생하기 전에 멈추는 것을 의미함.





### 7. Parameter Typing and Parameter Sharing

* 여러 파라미터가 있을 때 몇개의 파라미터를 공유하는 역할
* 특정 layer의 파라미터를 공유하거나 weight를 비슷하게 함
  * 각각의 네트워크에 파라미터 수를 줄이는 효과가 있다.
  * 파라미터가 줄어들면 일반적인 퍼포먼스가 증가하는 효과가 있어서 모델의 성능이 좋아지는데 도움이 된다.

* **Parameter Typing** : 파라미터 수를 줄이는 역할
  * 입력은 다른데 비슷한 작업을 하는 경우(ex. MNIST, SVHN dataset), 특정 레이어를 공유하거나 두개의 웨이트를 비슷하게 만든다
* **Parameter Sharing**
  * 같은 convolution filter가 전체 이미지를 모두 돌아다니면서 찍는 CNN이 있다.





### 8. Sparse Representations

* 어떤 아우풋의 대부분 '0'이 되길 원하는 것.
  * hidden layer가 나오면 그 값을 줄이는 패널티를 추가하면 sparse representation을 찾는데 도움이 될 수 있다.
* ex) one-hot-encoding
* **Sparse weights (L1 decay)**
  * 앞단의 행렬(네트워크의 weight)에 0이 많은 것
* **Sparse activations**
  * 뒷단의 행렬에 0이 많은 것을 더 중요하게 여김

* **ReLU**
  * 0보다 작은 activation은 0으로 바뀜
  * 아웃풋에 0이 많아지면 sparse activation할 수 있으므로 성능이 좋아진다.





### 9. Bagging and Other Ensemble Methods

* 앙상블 방법으로 알려진 model averaging method.
* 각각의 모델이 training set에서 같은 오류를 만들지 않기 때문에 정규화에 효과적이다.
* Variance : 예측에 대한 다양성
* Bias : 평균과 오차.. 정도로 생각할 수 있다.
* **Bagging** (Bootstrap Aggregation)
  * 데이터의 복원추출 후 여러 개의 표본을 만들어 이를 기반으로 각각의 모델을 개발한 후에 하나로 합쳐 1개의 모델을 만들어 내는 것.
  * **알고리즘의 안정성** : 여러개의 표본을 이용해 모집단을 잘 대표할 수 있게 된다.
  * 병렬처리를 사용할 수 있다.
    * 독립적인 데이터 셋으로 독립된 모델은 만들기에 모델 생성에 있어 효율적이다
  * ex) Random Forest
* **Boosting**
  * 틀린 케이스에 가중치를 줌으로써 이를 해결하는 것에 초점을 맞추는 모델
  * **정확성의 향상** : 오분류에 대한 높은 가중치를 부여해 이를 더 잘 해결할 수 있는 모델로 수정
  * 순차적인 모델의 학습
  * 오답에 높은 가중치를 주다보니 이상치에 취약할 수 있다.
  * ex) AdaBoost, xgBoost, GBM



* Bagging VS Boosting

  배깅과 부스팅은 모두 의사결정나무의 안정성을 높인다는 공통점이 있다. 

  하지만 치우침은 부스팅이 줄일 수 있고, 과적합 문제의 해결은 배깅만이 할 수 있다. 

  둘 다 표본추출에 있어서 데이터셋에서 복원 랜덤 추출하지만 부스팅은 가중치를 사용한다는 차이가 있다. 

  또한 배깅은 병렬적으로 모델을 만들지만, 부스팅은 하나의 모델을 만들어 그 결과로 다른 모델을 만들어 나간다. 즉 순차적으로 모델을 완성시켜 나간다.

  또한 가중치에 대해서도 배깅은 1/n으로 가중치를 주지만, 부스팅은 오차가 큰 개체에 대해 더 높은 가중치를 부여한다. 

  마지막으로 훈련 및 평가 항목에 대해서도 차이점이 있는데, 배깅은 트레이닝 셋을 만들어 그냥 계속 가지고 있는 반면, 부스팅은 트레이닝 셋을 만든 후에 업데이트 및 조정하는 과정이 추가가 된다.





### 10. Dropout

* Overfitting 문제를 해결하고자, 일반화 성능을 향상시키기 위한 방법
* 학습시 뉴런의 일부를 랜덤으로 '0'으로 만든다. 일종의 비활성화이며 이 비율은 조절이 가능하다.
* 확률은 일반적으로 'p' = 0.5를 사용하며, 학습할 때만 dropout하고 테스트할 때는 모든 뉴런을 사용한다.
* 이는 dropout을 통해 앙상블 학습처럼 마치 여러 모델을 학습시킨 것과 같은 효과를 주어 overfitting을 해결할 수 있다.





### 11. Adversarial Training

* 인간이 관측할 수 없는 노이즈를 넣어 완전히 다른 클래스의 데이터를 만든다.
* 입력은 조금 바뀔지라도 출력은 크게 달라지며, 그 때의 기울기가 매우 가파르다.
* 이는 overfitting이긴 한데, 많은 모델에 overfitting이 적용되기 때문에 성능이 잘나온다.
  overfitting을 숫자로 밀어버린듯??





#### 마치며...

오버피팅은 항상 발생한다! 어떤 문제를 풀던 발생하기 때문에 정규화를 항상 고려해야 한다...

앞으로 이와 관련된 내용은 추가로 알게 될 때마다 정리할 예정이다.

Overfitting의 해결은..

1. 1. 데이터를 늘린다.
   2. 샘플을 조금씩 늘려가면서 레이어를 늘린다 (예: 64 -> 128.. 512-> 1024...)
      데이터 수가 적을 땐 더욱 중요하다.
   3. Data augmentation을 잘 해야한다!
   4. 이 분야는 아트의 영역(?)이다. 감이 중요하기 때문에 문제를 풀어본 자만이 잘 알 수 있다...(a.k.a 노하우)

... 라고 한다. 열심히 해보자...