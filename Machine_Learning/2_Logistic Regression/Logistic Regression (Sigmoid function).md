## Logistic Regression using Sigmoid function

통계적 분류를 의해 linear regression을 사용하면 몇가지 문제점이 발생한다.

- 0~1 외의 y값을 갖게 된다.
- outliers에 민감하게 변한다.

위 두 문제점을 보여주는 예시는 다음과 같다.



<img src="https://user-images.githubusercontent.com/84625523/124118730-9a20c100-daac-11eb-9f7c-d980926d68cd.png" alt="image-20210627211127811"  />

![image-20210627211042176](https://user-images.githubusercontent.com/84625523/124118759-a6a51980-daac-11eb-8f13-c6d712d4de7c.png)

앞서 말했듯이 0과 1로만 구분되어야 하는데 그 외의 값을 갖게 된다. 
또한 가장 오른쪽에 있는 sample에 의해 regression line이 크게 변한 것을 확인할 수 있다.

### Sigmoid Function

이를 해결하기 위해 sigmoid function을 사용한다. sigmoid function의 식은 아래와 같다.

![sigmoid function](https://user-images.githubusercontent.com/84625523/124118817-b886bc80-daac-11eb-8d3c-5be0fcfb1ab4.png)

sigmoid function에서 e^(-x) x대신 (b0 + b1 * x1 + b2 * x2 + ... bn * xn)으로 바꿔 b0~n 값을 계산해서 예측하는 방식으로 진행된다.

sigmoid function의 양변에 자연로그(ln)을 씌워준 것을 logit transformation이라고 하고, 이를 통해 식을 linearlize할 수 있다. 양변에 자연로그를 씌워주면 아래와 같은 식이 된다.

![](https://user-images.githubusercontent.com/84625523/124119400-701bce80-daad-11eb-9f29-5cba8d943c7f.gif)

이처럼 함수의 모양이 linear한게 아니라 paramater들이 linear하게 표현하는 것을 linearlize한다 라고 표현한다.



### MLE - Maximum Likelihood Estimation

Logistic Regression의 parameter들을 예측하는 방법이다. likelihood function을 기반으로 하여 예측한다.

![](https://user-images.githubusercontent.com/84625523/124119800-ecaead00-daad-11eb-80b0-d9556e6ae07f.png)

위 식을 기반으로 𝛽를 예측하는 식은 다음과 같다.

![](https://user-images.githubusercontent.com/84625523/124120195-6fd00300-daae-11eb-8fba-ee0425222fc3.png)

x'은 x1~n을 의미한다. 이 값은 **Newton-Raphson** method로 구한다.





### 실습 코드_1

0. 라이브러리 import

   ```python
   import numpy as np
   from matplotlib import pyplot as plt
   from sklearn.linear_model import LogisticRegression
   ```

   

1. 데이터 불러오기

   ```python
   x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
   y1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
   
   x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
   y2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
   
   X = np.array([[0], [0.6], [1.1], [1.5], [1.8], [2.5], [3], [3.1], [3.9], [4], [4.9], [5], [5.1], 
                 [3], [3.8], [4.4], [5.2], [5.5], [6.5], [6], [6.1], [6.9], [7], [7.9], [8], [8.1]])
   y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
   ```

   이번 실습은 sigmoid function의 대략적인 예시이기 때문에 임의의 숫자를 이용했다.
   위에서 말했듯이 y값은 0 혹은 1로 고정되어 있고 분류를 하기 위한 regression을 진한다.



2. 모델 피팅 및 (x1, y1), (x2, y2) 그래프 확인

   ```python
   plt.plot(x1, y1, 'bo')  # (x1, y1) as blue dot
   plt.plot(x2, y2, 'ro')  # (x2, y2) as red dot
   
   model = LogisticRegression()
   model.fit(X, y)
   plt.show()
   ```

   <실행결과>
   ![image-20210627214143818](https://user-images.githubusercontent.com/84625523/124118871-ca685f80-daac-11eb-95c5-a200a7e67bc6.png)



3. b0, b1값 확인 및 sigmoid function 예측

   ```python
   print("b0 is : ", model.intercept_)
   print("b1 is : ", model.coef_)
   
   def logistic(classifier, x):
       return 1/(1+np.exp(-(model.intercept_ + model.coef_ * x)))
   
   for i in range(1, 120):
       plt.plot(i/10.0-2, logistic(model, i/10.0), 'go')	
   
   plt.axis([-2, 10, -0.5, 2])
   plt.show()
   ```

   - np.exp( num )은 num을 지수로 하는 e의 지수함수(e^num)으로 변환해주는 함수이다.
   - sigmoid function의 지수값과 맡게 parameter들을 넣은 것을 확인할 수 있다.
   - 이후 for문을 이용해 x값과 y값을 각각 i/10.0 - 2, sigmoid function의 결과값으로 넣어 plotting을 했다. 

   - x축은 [-2, 10], y축은 [-0.5, 2]의 범위를 갖는다

   <실행 결과>
   ![image-20210627215129724](https://user-images.githubusercontent.com/84625523/124118908-d48a5e00-daac-11eb-9416-6bc8417d641d.png)



4. x값에 대한 예측치

   ```python
   pred = model.predict_proba([[3.5]])
   print("prediction: ", pred)
   
   ## 실행 결과: prediction:  [[0.72860759 0.27139241]]
   ```

   - 이는 x가 3.5일 때 0일 가능성이 약 73%, 1일 가능성이 약 27%임을 의미한다.