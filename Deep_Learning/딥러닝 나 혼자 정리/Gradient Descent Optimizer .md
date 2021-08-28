# Gradient Descent Optimizer

### Gradient Descent

* loss function의 기울기(gradient)를 구하고, loss 값을 줄이는 방향으로 조정해 나가는 방법을 통해서 네트워크를 학습한다.
* 즉 현재 네트워크의 weight에서 내가 가진 데이터를 다 넣어서 계산한 전체 error에 미분을 하여 미분을 줄이는 방향을 알 수 있다.
* 그 방향으로 정해진 스텝량(learning rate, lr)을 곱해서 weight를 이동시킨다.
* 하지만 이는 시간과 computational force가 많이 투입되는 방법으로 개선된 **Stochastic gradient descent** 방법을 찾아낸다.



### Stochastic Gradient Descent

* batch size 만큼씩 training data를 분리해 훑고 지나가는 개념이다.
* GD의 경우는 모든 데이터를 가지고 weight를 최적화 한다면, SGD는 batch size만큼의 데이터를 이용해 최적화 하는 과정이다.
* 간단한 예시를 살펴보자.
  - GD
    - 모든 데이터를 계산한다 => 소요시간 1시간
    - 최적의 한스텝을 나아간다.
    - 6 스텝 * 1시간 = 6시간
    - 확실한데 너무 느리다.
  - SGD
    - 일부 데이터만 계산한다 => 소요시간 5분
    - 빠르게 전진한다.
    - 10 스텝 * 5분 => 50분
    - 조금 헤메지만 그래도 빠르게 간다!

* 하지만 이 역시 단점이 존재하는데 이는 다음과 같다.
  1. 방향설정이 일정하지 않다.
  2. learning rate의 최적화가 필요하다.
* 따라서 이보다 더 좋은 optimizer들이 존재한다.

<img src="https://user-images.githubusercontent.com/84625523/126278689-822f457c-50ed-497d-a9c7-048a0d505e11.png" style="zoom:80%;" />

### Other optimizers

* SGD에서의 문제점을 2가지 방향으로 해결한다.
  1. 방향설정이 일정하지 않다.
     * Momentum 이라는 변수를 설정해 해결한다.
     * 특히 local minima에 갇혔을 때 이를 빠져나오기 위해 사용한다.
     * SGD에서 계산된 접선의 기울기에 한 시점(step) 전의 접선의 기울기값을 일정 비율만큼 반영하여 언덕에서 공이 내려올 때, 중간에 작은 웅덩이가 있어도 빠져나올 수 있는 충분한 "관성"을 준다.
  2. learning rate의 최적화
     * 모든 매개변수에 동일한 learning rate를 적용하는 것은 비효율적이다.
     * Adagrad라는 알고리즘은 각 매개변수에 서로 다른 learning rate를 적용한다.
     * 처음에는 크게, 점점 작은 learning rate를 적용한다.



* 이후 Adagrad와 momentum의 두 방법을 섞은 Adam이라는 방법이 생겼고, 대부분 이 방법을 이용한다.



https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=lego7407&logNo=221681014509

https://seamless.tistory.com/38