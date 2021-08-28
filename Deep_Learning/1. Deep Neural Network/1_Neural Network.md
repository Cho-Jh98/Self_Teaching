## Neural Networks





** activation function = action potential and threshold



### Models

input → huge network of neurons → output

 <img src="https://user-images.githubusercontent.com/84625523/124887384-174db800-e010-11eb-95f0-7277bcca39bf.jpeg" alt="Layets" style="zoom:50%;">

1. Input layer. The first layer of neural network.
   * We keep feeding our network with data through the input layer
     * ex) digit image : Convert image to RGB value so every pixel will have a numerical value
       						     So we are going to have as many number of input neurons as pixel numbers.
2. Hidden layer
   * It is needed if we have problem that isn't linearly solvable. and this is most cases
   * More hidden layers 
     * may capture more information,
     * algorithm understands data better.
     * And this is why deep neural nets came to be.
   * If we have linearly separable problem, there is no hidden layer
3. Output layer
   * Result is in this layer



#### Overview

<img src="https://user-images.githubusercontent.com/84625523/124890036-993ee080-e012-11eb-9bb1-f2882eb59848.png" style="zoom:50%;" />

* Sum function
  * We add all the x's with corresponding w's.
  * equation:     ![Sum function](https://user-images.githubusercontent.com/84625523/124890528-1702ec00-e013-11eb-80cb-0fac20fd1b7d.png) 

* Weights
  * Parameter that can decide whether amplify(w>1) or deamplify(w<0)
  * This is exactly how the neural network can **learn**
  * Change(optimize) the edge weights until the neural net makes good prediction
* Activation function
  * It takes the output of the sum function and converts it
  * Step function, sigmoid function, ReLU activation function...
  * Introduce non-linearity to our model
  * Helps the network use the important information and suppress irrelevant data points





#### Activation function

**Review**

* Takes output of sum function and converts it

* step, sigmoid, ReLU..
* non-linearity
  * non-linear decision boundary 



##### Neurons perform a linear transformation on this input using the weights

* This is why we need activation function : To introduce non-linearity
* Many activation functions.



1. Step Function: it can ouput 0 or 1 accordingly

   <img src="https://user-images.githubusercontent.com/84625523/124892729-14a19180-e015-11eb-8c96-296a598b23c8.png" alt="Step function"/>

   * When input value is above or below a certain threshold, the neuron is activated and sends exactly same signal to the next layer
   * Problem
     * It cannot have multi-value outputs. Just 0 and 1
       So it cannot support classifying the input into one of several categories



2. Sigmoid function: it can ouput in the range [0,1]

   ![Sigmoid function](https://user-images.githubusercontent.com/84625523/124893405-aa3d2100-e015-11eb-9bc4-9902e6539ed4.png)

* Used for model where we have to predict the possibility as an output 
  * since probability of anything exists only between the range [0, 1]
* The function is differentiable during the training process
  * Optimization with gradient descent



3. Hyperbolic Tangent function: it can output in the range [-1, 1]

   <img src="https://user-images.githubusercontent.com/84625523/124894225-6c8cc800-e016-11eb-83f9-de24ea210130.png" alt="Hyperbolic Tangent function" style="zoom:50%;" />

* Differenciable.





### Big picture

<img src="https://user-images.githubusercontent.com/84625523/124887384-174db800-e010-11eb-95f0-7277bcca39bf.jpeg" alt="Layets" style="zoom: 33%;">

Intuition : If we change weight a little bit, than the output will change as well

we have tune the edge weight!!



* The neural network training alogrithm keeps changing the edge weight
  → for the given input, there will be right output in the output layer
  * We have a training dataset so we know correct outputs
  * Somehow we have to measure the accuracy and the error of the model



### Bias Unit

Sometimes we want to get not zero output even if the inputs are all zero.

Basically we can shift the activation function with bias unit

<img src="https://user-images.githubusercontent.com/84625523/124904026-8a125f80-e01f-11eb-92c3-ea53f88e7e3d.png" alt="Bias Unit illustration" style="zoom: 33%;" />

Here bias acts like a constant which helps the model to fit the given data

And formula of output with bias is

<img src="https://user-images.githubusercontent.com/84625523/124904060-94345e00-e01f-11eb-8396-21f75c5441fe.png" alt="Formula" style="zoom:33%;" />



## How to calculate Error-term?

First, we initialize the **w** edge weight of the neural network at random

* There will be some error between the predicted output and actual label

* The error can be calculated as **|Prediction - actual|**

* Then we construct so-called **loss-function / cost function**

  * These are nothing but the prediction error in the neural network model

* Formula of **loss-function** **L(w)** is

  ![Loss Function](https://user-images.githubusercontent.com/84625523/124905076-a236ae80-e020-11eb-9305-47da6ed54cfd.png)

  * L(w) = MSE loss function
  * y_i = actual value that we know from our training dataset
  * y_i' = prediction made by our artificial neural network

* L(w) transformed original problem to a quadratic optimization problem!

  * If L(w) is larget : there are huge error → neural network does not produce good results
  * If L(w) is low : there are small error → neural network is making good predictions

* So we have to find the **minimum** for the loss function



#### Minimize L(w)

* We have cost function with **w** dimensions that we have to find optimal w weights s.t. error is small  as possible.
* Several algorithms to find **min(L(w))**
  * Gradient Descent
  * Stochastic Gradient Descent
  * Meta-heuristic Approahes (genetic algorithms)
* Not an easy task to find optimal **w**
  * Huge number of **w** → high dimensionality
  * Search space is enormous when we have several hidden layers(especially in DL)



#### Gradient Calculation

* The **L(w)** measures the error so the smaller the better

  <img src="https://user-images.githubusercontent.com/84625523/124906872-7c120e00-e022-11eb-888a-37e583855a51.png" alt="Gradeint" style="zoom:50%;" />

1. Initialize **w** at random

   * It means that we start with a random position in the search space

2. Calculate the slope of the curve with partial derivative which is<img src="https://user-images.githubusercontent.com/84625523/124907076-b67bab00-e022-11eb-8847-b313ae8654e2.png" style="zoom: 25%;" />

   * we have fallow negative gradient to make **L(w)** smaller

3. Final formula

   ![Final Formula](https://user-images.githubusercontent.com/84625523/124907563-47528680-e023-11eb-95f5-831db55fdf29.png)

   * ![delta w_t](https://user-images.githubusercontent.com/84625523/124907813-8da7e580-e023-11eb-94a5-d7c0f24d48a6.png) : Change in edge weight at time **t**
   * **ɑ** : Learning rate
   * ![graident](https://user-images.githubusercontent.com/84625523/124907923-af08d180-e023-11eb-8691-3e55b75fc61f.png) : Gradeint (derivative of loss function)
   * **µ** : momentum
   * ![Previous change in edge weight](https://user-images.githubusercontent.com/84625523/124908032-d52e7180-e023-11eb-8065-4eccdb8b7394.png) : Previous change in **w** from previous iteration



* Learning rate(**ɑ**) : ~0.3
  * Define how fast our algorithm will learn
    * If it's too high : converges fast but not accurate
      						   may miss global optimum
    * If it's too low : Algorithm is slow but more accurate



* Momentum(**µ**) : ~0.6
  * Escape local minimum with this (not always works)
  * Define how much we relying on previous change by simply adds a fraction of the previous weight update to current one
    * High : helps to increase the speed of convergence of system
      		   but it can overshoot the minimum
    * Low : Cannot avoid local optimum and slows down the training





## Backpropagation

* Not an optimization algorithm
* It is the method that calculates the gradient of the **L(w)** with respect to **w**
  * It need differentiable activation function
  * Error layer **n** is dependent on the errors at the **n+1** layer which is the next one
    This is why it's called **back**propagation
* Error Flows Backward!!

* So if the activation function is sigmoid fuction(let's say y =f(x))

  > function sigmoid(x){
  >
  > ​		return 1/(1 + exp(-x))
  >
  > }
  > 
  >
  >
  > function dSigmoid(x){
  >
  > ​		return sigmoid(x) * (1 - sigmoid(x))
  >
  > }



### Formula

![Backpropagation formula](https://user-images.githubusercontent.com/84625523/125185349-62113f00-e25f-11eb-926e-5a3ad657159d.png)

* LHS : derivatibe of the **L(w)** with respect to edge weight
* 𝜹 : node delta at the next layer
* f(x*w) : activation of the neuron input



### Calculating the delta for the output layer 

### ((k+1)'th layer)



![delta formula](https://user-images.githubusercontent.com/84625523/125185501-1ad77e00-e260-11eb-9b33-f87977d39b1e.png)

* 𝛅_output : node delta at the output layer
* E : error term
  * difference between actual and predicted values 
* dSigmoid() : derivative of the activation function



### Calculating the delta for the hidden layer



![hidden layer](https://user-images.githubusercontent.com/84625523/125185580-9cc7a700-e260-11eb-8ef8-d8efd560e4c8.png)

* 𝛅_hidden : node delta at the hidden layer
* dSigmoid() : derivate of the activation function of the sum
* x_j * 𝛅_j : delta values of the previous(output) layer



So we have to know error term(E) to calculater 𝛅 at the output layer than use RHS of the first formula



#### mathmatic calculation..





1. forward propagation
   * We feed the neural network with the input and calculate the output activation value.

2. Back propagation
   * Then we update the edge weights with deltas starting from the output layer.



## Exact Derivation

Backpropagation : method for calculating the partial derivative of the **loss function**.

Updated edge weights are proportional to the partial derivative.

E = 1/2 \sum_k{t_k - a_k}^2



![exact derivation](https://user-images.githubusercontent.com/84625523/125186036-f9c45c80-e262-11eb-8530-4687e5a8b80d.png)



We have to apply the so-called cahin rule





























