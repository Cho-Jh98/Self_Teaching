# Convolutional Neural Networks



 ## Why we need different approach?



* Dense neurla network is working fine, but if there are 1,000 neurons in each layer, number of weights are increasing dramatically.
  * It will be super slow..
* ex) 32x32 pixel image → 1024 pixels → 1000s connections and weights → combinational explosion
  * plus we have colors too.
* Gradient descent changes the weights according to the learning rate and the gradient

### IT will be Super Slow!!



 ## Solution : Convolutional Neural Network!

### CNN : Convolutional Neural Network

* CNN has an assumption
  * Inputs are images!
  * We can encode certain properties into architecture!
* Neurons are not connected to every neuron with next layer : **Not a Dense network!**
* Under the hood : uses a standard neural network.
  * But : at beginning, it transforms the data to achieve the best accuracy possible.
  * ex) Self driving cars, pedestrain detection
  * Out performs Machine Learning such as SVM





## Theoratical background of CNN

### Main problem in ML and AI

#### ▶︎ Feature Selection!!

* What features to use to build our model?
* SVM, Basian, Densly Connected Neural network, CNN
  * All of them rely heavily on given dataset i.e given features
* How to choose best feature?
  * ex1) What features to use in order to recognize faces?
    * Location of nose, mouth, eyes ..
  * ex2) What features to use in order to recognize tiger?
    * Shape of ears, white and orange pattern..

#### CNN Finds the relevant feature by itself!





## 3 steps!

1. Convolutional operation
2. pooling
3. flattening





### 1. Convolutional operation

* Convolutional operation fromula

  <img src="https://user-images.githubusercontent.com/84625523/125633066-1e8b7609-3bb9-4fd9-8275-3619eaaaa0bd.png" style="zoom:33%;" />

  * In image processing

    * convolution is the process of adding each element of the image to its local neigbors, weighted by kernels
    * Image ( f(x) ) → matrix representation
    * kernel (= feature detector, filter) ( g(x) ) → another matrix

  * Convolution : matrix operation... We have to multiply values

  * Matrix operation formula

    <img src="https://user-images.githubusercontent.com/84625523/125633455-8a1bf539-930d-4dc2-9268-dd48dcfb4b9c.png" style="zoom:33%;" />



#### Kernel (Feature detector, filter)

* Feature detectors are represented as matirx
  * Helps detect the relevant features in a given image



ex1) **Sharpen Kernel** : Makes given image more sharp

* Matrix formula: 

  <img src="https://user-images.githubusercontent.com/84625523/125635235-139a7f92-e10b-4656-8d7a-47b59ecc6748.png" style="zoom:40%;" />

* Increase the pixel intensity of the given pixel

* reduce the pixel intensity of neigbor pixel

* Result: 

<img src="https://user-images.githubusercontent.com/84625523/125635330-9804ddb7-298c-4c8e-a402-e4c3ac4d0f5b.png" style="zoom:50%;" />

ex2) **Edge Detection Kernel** : Detect edges, somtimes can end up with relevent features

* Matrix formula : 

  <img src="https://user-images.githubusercontent.com/84625523/125635575-978fa9e6-23b8-4714-8cf7-178f82e71b20.png" style="zoom:40%;" />

* Decrease the pixel intensity of the given pixel

* X change the pixel intensity of neigbor pixel

* Result : 

  <img src="https://user-images.githubusercontent.com/84625523/125635565-538889f0-a5cb-4bd8-abd0-3a9c2cf71a34.png" style="zoom:50%;" />

  * Most important, widely used



ex3) **Blur Kernel** : Not that useful → X use for CNN

* formula : 

  <img src="https://user-images.githubusercontent.com/84625523/125636059-565c5e7a-2325-4c50-9688-870007cdca43.png" style="zoom:50%;" />

* Do not change pixel intensity of the given pixel

* Use neighbor pixel intensity value as well

* Result : 

  <img src="https://user-images.githubusercontent.com/84625523/125636046-e37c33ff-da47-455b-bbe5-b718705f0839.png" style="zoom:50%;" />



#### Every Kernel will use specific feature of image

* While using edge detector, we assume the edges are important
* But How to decide which feature detector to use?
  * By not deciding in advance
  * First, CNN uses many kernels.
  * Then, During the training process, it eventually selects the best possible





### Concrete example of Kernel

 <img src="https://user-images.githubusercontent.com/84625523/125639069-8759c571-1071-4e27-900a-e9fae9459fea.png"  />

1. Multiply Image with feature detector
2. Use several feature detector on the same image
3. Get lots of feature map (matrixes)
4. Use ReLU to every feature map in order to introduce some non-linearity



### Pooling operation

#### Spatial invariance

* Make sure to detect the same object no matter where is it located on the image
* Or whether it is rotated or transformed



#### Max pooling

* Select the most relevant features

  * This is how we deal with spatial invariance.
  * Just care about the most relevant features

  <img src="https://user-images.githubusercontent.com/84625523/125641491-628b5035-c1d2-41fa-bac2-6e56c9fd065a.png" style="zoom:33%;" />

  * Reduce the dimension of the image
    * To end with a dataset containing important pixel values without unnecessary noise
    * Plus: reduce number of parameter → reduce overfitting
  * Use the window, size of window is up to you.
  * Within the window, choose the maximum value = **Maximum pooling**
  * Windows should not overlap with each other
  * In our example, we are going to use 2x2 window
  * Result: 

  <img src="https://user-images.githubusercontent.com/84625523/125642655-0e5f1dcc-fb93-4e1c-802e-9c050abf6315.png" style="zoom: 80%;" />

* **Average pooling**

  * Instead of choosing maximum value, calculate the average of values in the window
  * Maximum is better for choosing most relevant feature.





So far so on...

<img src="https://user-images.githubusercontent.com/84625523/125643420-bf3e8d83-c1b1-4649-a50b-32d772384e09.png" style="zoom:67%;" />

* By using several feature detector (kernel, filter)
* apply ReLU
* apply max pooling
* We have as many layers of max pooling
  * Because we use max pooling on every single feature maps





#### Flatening

* The last operation. Flatening
* Transform matrix into one-dimentional vector
  * output of standard densely conected neural network

<img src="https://user-images.githubusercontent.com/84625523/125645833-52c11843-2038-4f1b-af3e-31ef95f8273d.png" style="zoom:67%;" />

* We want one-dimensional values to be an input value for neural network.
* Why is it good?
  * It's a preprocessing and use ANN with just the **most relevant feature values**
  * It doesn't store unnecessary values
  * Using multilayer neural network to learn non-linear combination of these important features.





#### Training

<img src="https://user-images.githubusercontent.com/84625523/125646479-c129d7f2-805a-4b6b-966f-254992e207cf.png"  />

* Use gradient descent (Backpropagation) as usual, as far as training the CNN are concerned
  * Update the edge weights according to the error + Choose the right filter
  * Change the edge weight and filters accordingly





### Data Augmentation

* We need a huge dataset for deep learining, But what if we don't?

* Solution : Data Augmentation

  1. Apply random transformations on the images

     ex) rotations, flipping, scaling...

     By transform, we can make several images
     → Lots of new dataset

  2. Beacuse of data augmentation, learning algorithm never uses same image twice

     → There is no overfitting

  3. Solves huge problem that there are no big dataset to train algorithm



## Example with mnist Data-set

0. library import

   ```python
   import matplotlib.pyplot as plt
   from keras.datasets import mnist
   from keras.models import Sequential
   from keras.layers import Dense, Dropout, Activation, Flatten
   from keras.layers.normalization import BatchNormalization
   from keras.utils import np_utils
   from keras.layers import Conv2D, MaxPooling2D
   from keras.preprocessing.image import ImageDataGenerator
   ```

   * From keras.models, we can import librarys for model design.
     * Dense : add densly connected layer
     * Dropout : regularization method. randomly set activation of given neuron to be 0. We can set the probability
     * Activation : Activation function. ex) 'relu', 'sigmoid', 'softmax'...
     * Flatten : make 1 dimentional array out of given dataset
     * Conv2D : adding convolutional network algorithm
     * MaxPooling2D : use max pooling method
   * BatchNormalization : Normalizing function
   



1. Load dataset

   ```python
   (X_train, y_train), (X_test, y_test) = mnist.load_data()
   
   print(X_train.shape)
   print(y_train.shape)
   print(X_test.shape)
   print(y_test.shape)
   ```

   * load dataset from dataset.mnist

   * 60,000 for training, 10,000 for test

     > <Result>
     >
     > (60000, 28, 28)
     >
     > (60000,) 
     >
     > (10000, 28, 28)
     >
     > (10000,)



2. Gray scalinng and show the image

   ```python
   plt.imshow(X_train[0], cmap='gray')
   plt.title("Class " + str(y_train[0]))
   ```

   > <Result>
   >
   > <img src="https://user-images.githubusercontent.com/84625523/126071622-70d33006-71c6-49ea-a229-30c67c4d6dd1.png" style="zoom:50%;" />



3. Format that tensorflow can handle : (batch, height, width, channel)

   ```python
   features_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
   features_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
   
   features_train = features_train.astype('float32')
   features_test = features_test.astype('float32')
   ```

   * mnist data-set has 28x28 pixel and black and white. So height and width is 28, channel is 1



4. Nomalization (sort of...)

   ```python
   features_test /= 255
   features_train /= 255
   ```

   * Similar to min-max normalizaitono
   * Transform the values within the range [0,1]



5. One-Hot-Encoding

   ```python
   targets_train = np_utils.to_categorical(y_train, 10)
   targets_test = np_utils.to_categorical(y_test, 10)
   ```

   0 → (1, 0, 0, ... ,0)
   1 → (0, 1, 0, ... , 0)

   ...

   9 → (0, 0, 0, ... ,1)



6. Build CNN model

   ```python
   model = Sequential()
   
   model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
   model.add(Activation('relu'))
   ```

   * Input shape → 28x28 pixel, 1 channel
   * 32 is number of filters



7. Build Network

   ```python
   model.add(BatchNormalization())
   model.add(Conv2D(32, (3, 3)))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   
   model.add(BatchNormalization())
   model.add(Conv2D(64, (3, 3)))
   model.add(Activation('relu'))
   
   model.add(BatchNormalization())
   model.add(Conv2D(64, (3, 3)))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   
   #flatten
   model.add(Flatten())
   model.add(BatchNormalization())
   
   # fully connected layer
   model.add(Dense(512))
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   
   # regularization avoid overfitting
   model.add(Dropout(0.2)) # probability is 0.2
   model.add(Dense(10, activation='softmax'))
   ```

   * Normalize the activation of previous layers after convolutional phase
   * Convolutional phase: Con2D & Activate
   * Trnasformation maintains the mean activation close to 0 &  standard deviation close to 1
     * scale of dimension remains the same
     * reduces running time of training siginificantly
   * Dropout is form of regularization



8. Compile the model

   ```python
   model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
   model.fit(features_train, targets_train, batch_size=128, epochs=2, validation_data=(features_test, targets_test), verbose=1)
   ```

   ><Result>
   >
   >```python
   >Epoch 1/2
   >469/469 [==============================] - 183s 346ms/step - loss: 0.1926 - accuracy: 0.9410 - val_loss: 1.2067 - val_accuracy: 0.6376
   >Epoch 2/2
   >469/469 [==============================] - 160s 342ms/step - loss: 0.0319 - accuracy: 0.9902 - val_loss: 0.0400 - val_accuracy: 0.9874
   ><keras.callbacks.History at 0x7f38dabcb310>
   >```



9. Print the score

   ```python
   score = model.evaluate(features_test, targets_test)
   print("test accuracy: %.2f" % score[1])
   ```

   ><Result>
   >
   >```python
   >313/313 [==============================] - 7s 24ms/step - loss: 0.0400 - accuracy: 0.9874
   >test accuracy: 0.99
   >```



10. Bonus: Data argumentation

    * It helps reduce overfitting.
    * Basically what it does is to rotate, flip, scale images.
      * Makes extra dataset, which is independent from its original dataset

    ```python
    train_generator = ImageDataGenerator(rotation_range =7, width_shift_range = 0.5, shear_range=0.22,
                                         height_shift_range = 0.07, zoom_range=0.05)
    test_generator = ImageDataGenerator() # don't want transformation for test-set
    
    train_generator = train_generator.flow(features_train, targets_train, batch_size = 64)
    test_generator = test_generator.flow(features_test, targets_test, batch_size = 64)
    
    model.fit_generator(train_generator, steps_per_epoch = 60000/64, epochs = 10,
                        validation_data=test_generator, validation_steps=10000/64)
    ```

    > <Result>
    >
    > ```python
    > Epoch 1/10
    > /usr/local/lib/python3.7/dist-packages/keras/engine/training.py:1915: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
    >   warnings.warn('`Model.fit_generator` is deprecated and '
    > 937/937 [==============================] - 187s 198ms/step - loss: 0.2771 - accuracy: 0.9142 - val_loss: 0.0387 - val_accuracy: 0.9874
    > Epoch 2/10
    > 937/937 [==============================] - 184s 196ms/step - loss: 0.1489 - accuracy: 0.9529 - val_loss: 0.0434 - val_accuracy: 0.9884
    > Epoch 3/10
    > 937/937 [==============================] - 184s 196ms/step - loss: 0.1242 - accuracy: 0.9610 - val_loss: 0.0348 - val_accuracy: 0.9890
    > Epoch 4/10
    > 937/937 [==============================] - 184s 196ms/step - loss: 0.1141 - accuracy: 0.9647 - val_loss: 0.0268 - val_accuracy: 0.9912
    > Epoch 5/10
    > 937/937 [==============================] - 184s 197ms/step - loss: 0.1060 - accuracy: 0.9669 - val_loss: 0.0290 - val_accuracy: 0.9910
    > Epoch 6/10
    > 937/937 [==============================] - 183s 195ms/step - loss: 0.0987 - accuracy: 0.9691 - val_loss: 0.0234 - val_accuracy: 0.9919
    > Epoch 7/10
    > 937/937 [==============================] - 183s 195ms/step - loss: 0.0907 - accuracy: 0.9711 - val_loss: 0.0204 - val_accuracy: 0.9937
    > Epoch 8/10
    > 937/937 [==============================] - 183s 195ms/step - loss: 0.0886 - accuracy: 0.9719 - val_loss: 0.0196 - val_accuracy: 0.9945
    > Epoch 9/10
    > 937/937 [==============================] - 183s 195ms/step - loss: 0.0867 - accuracy: 0.9730 - val_loss: 0.0208 - val_accuracy: 0.9928
    > Epoch 10/10
    > 937/937 [==============================] - 183s 195ms/step - loss: 0.0809 - accuracy: 0.9744 - val_loss: 0.0293 - val_accuracy: 0.9915
    > <keras.callbacks.History at 0x7f38dac45690>
    > ```





### What do I have to learn more is...

* Batch Normalization
  * https://eehoeskrap.tistory.com/430
* Regularization except dropout and data augmentation.
  * https://wegonnamakeit.tistory.com/9
* Not for CNN(maybe) but.. Optimizers
  * https://seamless.tistory.com/38
