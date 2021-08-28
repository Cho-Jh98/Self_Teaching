# CNN with cifar10 dataset



0. Library import

   ```python
   from keras.datasets import cifar10
   import numpy as np
   from keras.models import Sequential
   from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
   from keras.layers.normalization import BatchNormalization
   from tensorflow.keras.utils import to_categorical
   from keras.optimizers import SGD
   from keras.preprocessing.image import ImageDataGenerator
   ```



1. Load dataset

   ```python
   (X_train, y_train), (X_test, y_test) = cifar10.load_data()
   
   print(X_train.shape)
   print(X_test.shape)
   print(y_test.shape)
   print(np.unique(y_train))
   ```

   * 50,000 for training, 10,000 for testing
   * 32x32 pixel images, 10 output classes
   * It will take some time to download the dataset.

   > <Result>
   >
   > (50000, 32, 32, 3) 
   >
   > (10000, 32, 32, 3) 
   >
   > (10000, 1) 
   >
   > [0 1 2 3 4 5 6 7 8 9]



2. Preprocessing the data

   ```python
   y_train = to_categorical(y_train)
   y_test = to_categorical(y_test)
   
   print(y_train.shape)
   print(y_train)
   print(y_test.shape)
   print(y_test)
   ```

   ><Result>
   >
   >(50000, 10)
   >
   > [[0. 0. 0. ... 0. 0. 0.]
   >  [0. 0. 0. ... 0. 0. 1.]
   >  [0. 0. 0. ... 0. 0. 1.]
   > ... 
   > [0. 0. 0. ... 0. 0. 1.]
   > [0. 1. 0. ... 0. 0. 0.]
   > [0. 1. 0. ... 0. 0. 0.]]
   > (10000, 10)
   >[[0. 0. 0. ... 0. 0. 0.]
   > [0. 0. 0. ... 0. 1. 0.]
   > [0. 0. 0. ... 0. 1. 0.]
   > ... 
   >[0. 0. 0. ... 0. 0. 0.]
   >[0. 1. 0. ... 0. 0. 0.]
   >[0. 0. 0. ... 1. 0. 0.]]



3. Normalize the data

   ```python
   X_train = X_train/255.0
   X_test = X_test/255.0
   ```



4. Build a model and design a layer

   ```python
   model = Sequential()
   
   model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same', input_shape = (32,32,3)))
   model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
   model.add(MaxPooling2D(2, 2))
   model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
   model.add(MaxPooling2D(2, 2))
   model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
   model.add(MaxPooling2D(2, 2))
   model.add(Flatten())
   model.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
   model.add(Dense(10, activation='softmax'))
   ```

   * kernel_initializer : random weight initializer
     * he_unifrom : uniform distribution of weight.



5. Set optimizer and compile the model

   ```
   optimizer = SGD(learning_rate=0.001, momentum = 0.95)
   
   model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
   
   history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)
   ```

   > <Result>
   >
   > ```python
   > Epoch 1/50
   > 782/782 - 229s - loss: 1.7036 - accuracy: 0.3839 - val_loss: 1.4237 - val_accuracy: 0.4863
   > Epoch 2/50
   > 782/782 - 215s - loss: 1.2990 - accuracy: 0.5367 - val_loss: 1.1774 - val_accuracy: 0.5831
   > Epoch 3/50
   > 782/782 - 215s - loss: 1.1259 - accuracy: 0.6032 - val_loss: 1.0882 - val_accuracy: 0.6157
   > Epoch 4/50
   > 782/782 - 216s - loss: 0.9991 - accuracy: 0.6503 - val_loss: 1.0321 - val_accuracy: 0.6403
   > 
   > …
   > 
   > Epoch 46/50
   > 782/782 - 220s - loss: 3.2107e-04 - accuracy: 1.0000 - val_loss: 2.4205 - val_accuracy: 0.7379
   > Epoch 47/50
   > 782/782 - 220s - loss: 3.0506e-04 - accuracy: 1.0000 - val_loss: 2.4324 - val_accuracy: 0.7360
   > Epoch 48/50
   > 782/782 - 221s - loss: 2.9199e-04 - accuracy: 1.0000 - val_loss: 2.4458 - val_accuracy: 0.7368
   > Epoch 49/50
   > 782/782 - 223s - loss: 2.7807e-04 - accuracy: 1.0000 - val_loss: 2.4565 - val_accuracy: 0.7371
   > Epoch 50/50
   > ```
   >
   > * Too slow and Low accuracy.
   > * We should introduce Normalization to make it faster and more accurate

6. Test the model

   ```python
   model_result = model.evaluate(X_test, y_test, verbose = 0)
   print("Accuracy of CNN model: %s" % (model.result[1] * 100.0))
   ```

   





### Add Regularization to CNN with cifar10 dataset

Change the 4. of layer design to this

```python
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same', input_shape = (32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same')))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
```

* We added BatchNormalization and Dropout

* Accuracy  = 83%
  * Much better result than without regularization



* Can be better with data augmentation















