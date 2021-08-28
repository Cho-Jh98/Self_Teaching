# Recurrent Neural Networks



* Google translator relies on RNN
* RNN to make time series analysis!



#### Turing test

* Computer passes the Turing test if a human is unable to distinguish the computer from human in blind test
  * RNN are able to pass this test
    : Well trained recurrent network is able to understand English for example.
  * Learn Language models!!

* Make sure that the network can learn connections in the data even when they are far away from each other.
  * RNN can deal with relationships that are far away from each other

* Image descriptions with hibrid approach ( CNN + RNN )



## What is Recurrent Neural Network?

* For start, Deep neural network make independent prediction.
  * For example, training examples are independent of each other; tiger, elephants, rabbit...
  * So the prediction is independent
  * **p(t)** is not correlated with **p(t-1)** or **p(t-2)** and so on..
* But RNN's prediction is dependent from previous predictions!!
  * Training examples are **Correlated**
  * So **p(t)** depends on **p(t-1)**, **p(t-2)** and so on..
  * So RNN can predict the next word(picture) in given sentence. Or stock prices tmr.
    * This is important in natural language processing



## Architecture of RNN

<img src="https://user-images.githubusercontent.com/84625523/126074108-f5baf301-0d36-4c70-b593-3e200c8c112d.png" style="zoom: 33%;" />

* First of all untill now we had multilayer feedforward neural network.
  * Input, hidden, output layer
  * every layer can have several neurons

* RNN representation squeeze layers

  * Every single layer has one node
  * But, every node represent givien layer that can contain several neuron
  * ex) Middle node in RNN representation have all of information of 3 node in hidden layer of multilayer

* Hidden layer in RNN is reconnected to itself

  * Hidden layer gives an ouput & feeds back to itself

  * It will look like below figure.

    <img src="https://user-images.githubusercontent.com/84625523/126074239-7e9d82ae-db3d-40a1-80c2-72f48d00ecc2.png" style="zoom:33%;" />





### Formula for Hidden layer in RNN

* figure

  <img src="https://user-images.githubusercontent.com/84625523/126074440-5903c9a0-c34a-4998-98ee-4dca258b6503.png" style="zoom:33%;" />

  **x** : input
  **h** : activation after applying the activation function on the output

* Formula

  * Activation function

    <img src="https://user-images.githubusercontent.com/84625523/126074278-5e9aa773-35d2-48f7-9e59-1dde3123e3f0.png" style="zoom:50%;" />

    * activation
    * g(x) : activation function
    * W : weight
    * b : bias

  * Output function

    <img src="https://user-images.githubusercontent.com/84625523/126074484-a9e6b9bd-f0bd-44a7-ba50-2535567448b8.png" style="zoom:45%;" />

    * **y** : outputt
    * g(x) : activation function
    * W : weight
    * b : bias



#### So the overall network would look like..



<img src="https://user-images.githubusercontent.com/84625523/126074542-4d919595-5c55-4d66-8a6e-2a67687a51e6.png" style="zoom:40%;" />

* So how can we train RNN?
  * Unroll it in time to end up with a standard feedforward neural network (such as gradient descent)
  * Already knows how to do.
* As we can see in overal picture of RNN, W_R is shared accross every single layer
  * for a feed forward network, these weights are different





## Vanishing/Exploding gradient problem



* When dealing with backpropagation, we have to calculate the gradient.

* When we asign loss function to RNN

  <img src="https://user-images.githubusercontent.com/84625523/126074902-4e1b7ade-3fbb-4a82-8297-b948437597f8.png" style="zoom:40%;" />

  * and the derivative will be

    <img src="https://user-images.githubusercontent.com/84625523/126074857-e24fb024-7b41-4a63-ba5e-a00e30b550ab.png" style="zoom:50%;" />

  * So we just have to apply chain rule several time to find derivative of loss funciton

    <img src="https://user-images.githubusercontent.com/84625523/126074942-edd4540e-b942-4132-b960-431ed6909924.png" style="zoom:50%;" />

* When we multiply the weights several times..
  * When x < 1 : the result will be smaller and smaller
    * Leads to **VANISHING GRADIENT PROBLEM**
  * When x > 1 : the result will be smaller and smaller
    * Leads to **EXPLODING GRADIENT PROBLEM**
* By the way we use **Backpropagation Through Time(BPTT)**
  * Same as backpropagation
  * But gradients/error signal will also flow backward from future time-steps to current time-step



### Why is it a problem?

* If gradient become too small (vanishing gradient), it will be difficult for a model to train long-range dependencies

  * We want our model to find relationshipt that's far away from each other.

* For RNN, local optima are much more significant problem than with feed-forward neural network

  * Error function surface is quite complex

    <img src="https://user-images.githubusercontent.com/84625523/126075220-26adc98f-68a9-4d0f-9fc3-97f770e77e15.png" style="zoom:33%;" />

  * Thes complex surfaces have several local optima. But, we want to find global one

    * Use **meta-heuristic** approaches





### How can we solve this problem?



#### Exploding Gradient Problem

* Truncated BPTT algorithm
  * simple backpropagation, but we do only backpropagation through **k** time-step
* Adjust the learning rate with **RMSProp** (adaptive algorithm)
  * We normalize the gradient : use moving average over the root mean and squared gradients



#### Vanishing Gradient Problem

* Initiate the weights properly (Xavier-initiation)
* Proper activation functions such as ReLU function
* Use other archiectures : LSTM or GRUs





## Long Short Term Memory (LSTM)

### Gates?

* Instead of the **h** units
  * Add some memory to the neural network + manipulate these memory cells
    * Flush the memory - **Forget** **Gate**
    * Add to memory - **Input** **Gate**
    * read from memory - **Output Gate**



### Output Gate

* Output gate determines the **h** output based on the previous steps stored in memory

  * Illustration

  <img src="https://user-images.githubusercontent.com/84625523/126169244-5d37138a-b2dd-4f61-8ac9-66e241a5cae0.png" style="zoom:33%;" />

  * Lots of information in the memory

  * Output gate determine what data stored in memory is important or not

    * ex) Look at given image → Finds relevant feature (shape of ear) →
      	  When next iteration happens, the algorithm forgets first feature

      Store the shape of ear as relevant feature in memory in order to make sure in further iteration, we remember important information

  * Algorithm is going to 'remember' the most important features.

* It's important to use sigmoid activation function for output gate → Transform the values within the range [0,1]

  * Can manipulate the values in the memory

  * Formula

    <img src="https://user-images.githubusercontent.com/84625523/126170054-9f0dddda-8756-4004-b753-e4d2c25295d0.png" style="zoom:40%;" />

  * Illustration with formula

    <img src="https://user-images.githubusercontent.com/84625523/126170325-07dd9407-6ec0-44f1-8a03-3a1845e0eeb5.png" style="zoom:40%;" />

    * 0 → We do not care about the information present in the memory
      * No effect on training procedure
    * 1 → We take all the information present in the memory
      * Curious about the information present in memory cell





### Forget Gate

* Illustration

  <img src="https://user-images.githubusercontent.com/84625523/126171043-db9a2262-2141-4640-aedf-013667ed4b8b.png" style="zoom:40%;" />

  * With forget gate we can manipulate the **content of the memory**
    * **Not the output!!**
  * 0 → Get rid of the information present in the memory
  * 1 → Keep all the information present in the memory

* Use sigmoid activation function!!!
  * If tanh activation function is used, the result will range within [-1, 1] → It won't work

* Edge weight btw **output gate** and **forget gate** is different
  * Optimal weight for ouput gate, forget gate and input gate also.





### Input gate

* With input gate, we can write new data to the memory and with another gate, we can control what to keep

* Illustration

  <img src="https://user-images.githubusercontent.com/84625523/126171924-6ac907d2-62ca-402c-b817-7c9c012e69d3.png" style="zoom:35%;" />

  * This is complete picture of all the gates.
  * For input we can use sigmoid or ReLU, but usually we use tanh
  * By using sigmoid activation function in input gate, we can control what to keep from the input
    * because there's lots of irrelevent features (ex. background, loacation...)



#### Problem!!

* Complexity of the model..
  * Lots of weights to update
  * Too slow!





## Gated Recurrent Units(GRU)



* These units are a simplified **LSTM** blocks
  * All the gates are included in a single update gate
* Why to use GRU?
  * To cope with vanishing gradient problems!!
* GRU controls the flow of information like LSTM, but without having to use a memory unit
  * Just exposes the full hidden content without any control
* It's a quite simple model
  * More efficient achieved by simple model



*  Illustration

  <img src="https://user-images.githubusercontent.com/84625523/126173304-55541472-616c-41e0-93e6-5c97f0c34d90.png" style="zoom:33%;" />

  * No memory cell at all.
  * **Update** **gate** : Controls input itself
    * Determine which input to use and whats not
  * Remember gate : Controls how much previous steps inpact on new step
  * Input gate
