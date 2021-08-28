# Deep Neural Network theory



## what we are going to learn?

* Topology

* Role of 

  * activation function

  * Loss function

  * hyperparameter

    * Learning rate, momentum...

    

### Topology

<img src="https://user-images.githubusercontent.com/84625523/125196244-14fb9000-e294-11eb-9799-dbefd32723c9.png" alt="Topology" style="zoom: 33%;" />

* Every node is connected to every node in the next layer
  * There are lots of weights in such a network
* A little change in **w** results in change in the outout!!
* Training a network means adjusting the edge weights in the layer
* There are several hyperparamenters we can tune
  * ex) learning rate, momentum...



#### So How to calculate the activation?

$$
\text{Activation = }\sum_{i=0}^nx_iw_i
$$

* Linear combination of inputs and edge weights.
* introduce non-linearity
  * use activation function ex) sigmoid, relu..



### Deep Neural Network

Deep learning means we have several hidden layers : Usually 5~10 hidden layers
→ Other promblems may arise.

* Choose the activation function very carefully
  * sigmoid is not going to be best solution possible.
  * optimization method need to be chosed carefully



#### Activation function!!

**Role : Make neural network Non-Linear!!**

* w/o activation function, the network is just a linear transformation, which is not strong enough to make many kinds of data.
* Adding more parameters to the model instead of using activation function slows training.
* Types of Activation function
  * linear activation function
  * sigmoid activation function
  * tanh activation function
  * ReLU activation function
    * Rectified Linear Unit activation function
  * leaky ReLU activation function



1. Linear Activation Function.

   * **f(x) = x**

     <img src="https://user-images.githubusercontent.com/84625523/125196762-f9918480-e295-11eb-961a-537510be1558.png" style="zoom:33%;" />

* Identity operator : function passes the signal through unchanged
* Usually do not change the input when dealing with input layer
  * So we can say that the **input layer** has linear activation function
  * Input data doen't change the data at all



2.  Sigmoid activation function

   <img src="https://user-images.githubusercontent.com/84625523/125196839-512ff000-e296-11eb-91ae-3f8bf855e65b.png" alt="Sigmoid" style="zoom:33%;" />

* It reduces extreme values and outliers in the data without removing them
  * sigmoid fucntion transforms data in the rage [0, 1]
  * interpret the results as probabilities
    * prefer logistic to linear regression(we can deal with probabilities)
  * outputs an independent probability for each class



3. tanh activation function

<img src="https://user-images.githubusercontent.com/84625523/125196990-e59a5280-e296-11eb-8058-4a1b43b9a367.png" alt="tanh" style="zoom:33%;" />

* similar to the sigmoid function but the range is [-1, 1]
* It can handle negative values as well



4. ReLU activation function

<img src="https://user-images.githubusercontent.com/84625523/125197038-22fee000-e297-11eb-8cee-93e0633b8d23.png" style="zoom:33%;" />

* Rectified Linear Unit activation function

  * Activates only if the input is above a certain quantity

* **f(x) = max(0, x)** : ReLU activation function

* The most popular function

  * gradient is either zero or constant.
  * It can solve the vanishing gradient issue!

* Training procedure relies on the derivative of the activation function

* Each of the neural network's weights receives an update proportional to the gradient of the error function witth respect to the current weight in each iteration of training

* Problem

  * In some cases, gradients are vanishingly small → effectively prevent the weight from changing.

    → May lead to complete stop of neural network from further training.



5. softmax function
   * Generalization of the logistic function

<img src="https://user-images.githubusercontent.com/84625523/125197482-cbfa0a80-e298-11eb-870d-d22841e9e785.png" style="zoom:33%;" />

* Transform the value in the range [0, 1] that add up to 1
* Softmax function is used in various multiclass clasification methods
  * ex) digit classification
* Use softmax function in the last layer : want to classify the samples.
  * Choose the class with highes probability



### Loss function

* loss function measures how close the given neural network is to the ideal toward which it is training
  * We can calculate a value based on the error we observe in the network's prediction
* So we want to find optimal bias values and weights that will minimize the loss function.
  * Use gradient descent algorithm for the optimization

* Basically it's optimization problem



#### Types of loss function

1. MSE(Mean Squared Error)

* When we are dealing with regression ~ we only have one output feature

  

  ![](https://user-images.githubusercontent.com/84625523/125197802-10d27100-e29a-11eb-848c-3406be029c29.gif)

  * y' = prediction made by our artificial neural network
  * y = actual value that we know form our training dataset



2. Negative log likelyhood

* When dealing with classification

  * In this case there are several output values: digits when we calssify **MNIST** dataset

* Logarithm function monotonically increases

  * So minimizing the negative log likelyhood is same as maximizing probability
  * ex) Python optimize() method finds minimum, not the maximum of given function

  ![](https://user-images.githubusercontent.com/84625523/125198079-5a6f8b80-e29b-11eb-811b-d314feb6ce26.gif)

  * M : number of classes
    * 10 for handwritten digit classification
  * N : number of samples in the dataset





### Gradient Descent

<img src="https://user-images.githubusercontent.com/84625523/125198634-aa4f5200-e29d-11eb-9c24-380039ef75ee.png" style="zoom:33%;" />

* **Higher region**: the network makes lots of mistakes, because the **L(w)**'s value is still big
* **Lower region**: the network makes good prediction, because the **L(w)**'s value is small



1. Initialize the **w** weights at random at begining
   * starting from a given point in this landscape(figure) with given **w** and **L(w)** value

2. Negative gradient is pointing in the direction of lowest point

3. Just fallow the gradient

4. Repeat until convergence

   convergence : ![](https://user-images.githubusercontent.com/84625523/125198806-64df5480-e29e-11eb-84b6-4d1d1823ac7d.gif)



* Gradient descent works fine for convex loss function(with one minimum), but not all the time.

  * For example, look at loss function below.

  <img src="https://user-images.githubusercontent.com/84625523/125199494-63635b80-e2a1-11eb-8709-2903df5f8d9f.png" style="zoom:33%;" />

  * This loss function has local minimum and global minimum.

  * If gradient descent reaches local minimum,  it is trapped and it is one of the drawback.

* Solution : Use genetic algorithms or Simulated annealing

* Btw normalizing the original dataset is usually helpful
  * min-max normlization or z-score normalization
  *  X  →  ![](https://user-images.githubusercontent.com/84625523/125199416-11bad100-e2a1-11eb-9e09-3ad7a5839f57.gif)
  * Normalization makes sure gradient descent will converge faster with more accuracy



#### Stochastic gradient descent

* Update the gradeint and parameter vector after every single training sample.
  * If we use subset of original dataset, it is called minibatch stochastic gradient descent
* Compare with gradient descent
  * Faster convergence since using less data
  * Hence not that accurate
  * Do not always converges to the same minimum while gradient descent does.
    * former is called stochastic, latter one is called deterministic



## Hyperparameter



We can tune our neural network with different parameters such as 'learning rate', 'momentum' and so on.

* Want to avoid Overfitting as well as Underfitting
* Want to train algorithm as quick as possible.



### Learning Rate

![Learning Rate](https://user-images.githubusercontent.com/84625523/125198806-64df5480-e29e-11eb-84b6-4d1d1823ac7d.gif)

* How we define the pace of learning
  * Coefficient **ɑ** when we are dealing with the gradient.

* **ɑ > 1** 
  * Learning rate is too high
  * Training will be fast, but it may miss the optimum
* **ɑ < 0.01**
  * Learning rate is too small
  * Training will be slow, but it will find the optimum
  * The optimum could be local or global optimum



### Momentum

* Helps learning algorithm get out of spots in search space (local minimum) where it would otherwise become stuck.

  * Value between [0, 1] that increases the size of the steps taken towards the minimum by trying to jump from local minima.

  * Momentum is large : 
    * learning rate should kept smaller
    * convergence will happen fast





### Regularization



#### Common problem

* Learning algorithm works fine (with high accuracy) on training dataset, but not able to make good prediction on test dataset
  * Model is not able to grasp the relevant relationship between the features
  * It's called overfitting



#### Solution

1. **Weight decay**

   * Reduces over-fitting, improves the performance of the algorithm on the test set
   * Aim is to prevent edge weights getting too large

   1) **L1 Regularization** : sum of absolute weights

   2) **L2 REgularization** : sum of squared weights

   * Technically it means that we add some additional value to **L(w)** loss function

     Formula

     > <img src="https://user-images.githubusercontent.com/84625523/125824055-c3624a35-7247-4748-a059-171c078da93e.png" style="zoom:50%;" />



2.  **Dropout**

* **DROPOUT** is inexpensive regularization mothod.
  * temporarily, we set the activation of given neuron to be 0
    * means get rid of some neurons
  * works well with stochastic gradient descent method
  * uses in the hidden layer exclusively.
  * omit neurons with **p**, probability
* Prevent coadaptation among detectors, which helps drive better generalization in given models
* If traning records rises up, it becomes less effective.


