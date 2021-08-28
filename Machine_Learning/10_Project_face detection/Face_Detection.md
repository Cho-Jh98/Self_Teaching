# Computer Vison





## Computer vision

#### Datasets

* MNIST Handwritten digit dataset
  * 28x28 pixels images
  * We already know there are 10 output classes
* Olivetti Faces dataset (Not a detection, **Recognition**)
  * 64x64 pixels images
  * We know it is faces



#### Computer vision : We are after general solution

Given image → Make sure the alogrithm is able to **detect** and **recognize** objects(faces, cars, dogs)

 

### Viola-Jones Algorithm

:  Tries to find most relevent features for human face (when we are dealing with face detection)

So, what are the most relevent features for human face?

→ two eyes, nose, lips, forehead...

* Construct an algorithm based on the most relevent features

  → if the algorithm does not find one of these features, 
       It comes to conclusion that there is no human face on that region of the image



#### Feature of this algorithm

* This algorithm handles grayscale Image(white and black) 
  * So the first step of algorithm is convert image into grayscale

* It need training and test datassets
* Needs positive images and negative images
  * Positive images are images of face for face detection
  * Negative images are everything but face images.
  * This is how the algorithm learns the **most relevent features**



* shift the window with overlap and find the relevent features
  * find 2 eyes, nose and mouse



#### Difficulty

* Size of the face may differ
  * window size is too small to include all the relevent features of face
    → algorithm won't work fine
  * some face can be closer to camera
    → appear bigger than other faces in the background
* Using **scaleFactor** in **OpenCV** compensates for issues above
* Training uses 24x24 image so we have to rescale the input image
  → resize a larger face to a smaller one



## Harr-Features

#### Haar-wavelet is a sequance of rescaled square-shaped "Functions"

~ very similar to Fourier-analysis and convolutional kernels
**Haar features are the relevent features for face detection**



### Edge features and Line features

<img src="https://user-images.githubusercontent.com/84625523/124713377-6d9cf700-df3b-11eb-89a5-210542373fb7.png" style="zoom:50%;" />



### Mathmatics in Haar-feature



<img src="https://user-images.githubusercontent.com/84625523/124714718-1566f480-df3d-11eb-9ba5-74a4036250ea.png" alt="Harr-feature illustration" style="zoom:50%;" />

* Viola-Jones algorithm will compare how close the real scenario is to the ideal case

  1. Sum up the white pixel intensities

  2. Sum up the dark pixel intensities

  3. And 𝚫 is the difference between mean of dark pixel intensities and that of white pixel intensites

  ![equation](https://user-images.githubusercontent.com/84625523/124715112-91f9d300-df3d-11eb-9e10-344f4abb723a.png)

  * 𝚫 for ideal Haar-feature is 1

  * 𝚫 for the real image is 0.74 - 0.18 = 0.56

* The closer the value to 1, the more likely we have found a **Haar-feature**

  * We are never going to get 0 or 1 because there is no completely black or white.



## Integral Images

**Problem** : We have to caculate the average of a given region several times.
				   It means we have to use Haar-feature with all possible size and location
				   which is up to 200k features.

​				   The time complexity of this operations are O(N^2)

**Solution** : Use Integral image apporach to achieve O(1).
				   **Every time we have to use quadratic algorithm > we need better solution**



#### Illustration

<img src="https://user-images.githubusercontent.com/84625523/124717154-cff7f680-df3f-11eb-9b1d-68ddbcd6f112.png" alt="Integral Image" style="zoom:50%;" />



Each element of integral image : sum of all of left and above element.

ex) If we want to calculate sum of 3x2 in the center of matrix

<img src="https://user-images.githubusercontent.com/84625523/124717629-4694f400-df40-11eb-896d-5de98a632352.png" style="zoom:33%;" />

Sum = 3.7 - 1.7 - 0.5 + 0.2

By integral image, we can achieve O(1) time complexity for handling Haar-features
→ we assume these features are rectangles





## Computer Vision - Boosting



* Can boost some part of the algorithm with the help of integral image approach

* But **Too Many Features!!!**

  * Most of the features are even irrelevent and not important at all

  Solution Is **Boosting!!**



### Boosting for computer vision

* Finding a H(x) model which is a strong classifier

  * Keep combining h(x) which is a weak learner (with a single Haar-feature)
  * Every h(x) weak learner make a prediction based on a single Haar feature

* Formula:

  <img src="https://user-images.githubusercontent.com/84625523/124777858-744b5e80-df7b-11eb-9d3e-584ccbf102c6.png" alt="Boosting Formula" style="zoom:125%;" />

  * At beginning, all h(x) has same weight.

    ~ all of them contribute to the final decision(face is detected or not.

  * During the training phase of **Viola-Jones Algorithm**
    * weight for the h(x) is updated (h(x) : weak learners / the features)
    * finally we have the relevent feature with higher weights.
  * Look at boosting chapter again...





* First weak learner detect some of the images correctly But with some mis-classification
* generate new weak learner that focus on mis-classifed image
  * Find a new feature that can classify mis-classified image
* Do it over and over again until we have strong classification



## Cascading

With help of boosting, we can make algorithm quite fast. But we can do BETTER!!



#### flow of thought..

1. Most of the image is non-face region
2. It's better to have a simple method to check if a window  is not a face region
   * If it's not, discard it in a single shot
3. Do not process the unnecessary region
   * focus on region where there can be a face (which we did not checked yet) instead

→**This is why we use the cascade classifier concept**

So instead of using all the feature, we only use the most relevant ones in the first iteration.
(first stage contain very less features...)



#### How to find most relevant feature?

Use a boosting algorithm!!

<img src="https://user-images.githubusercontent.com/84625523/124777858-744b5e80-df7b-11eb-9d3e-584ccbf102c6.png" alt="Boosting Formula" style="zoom:125%;" />

The h(x) classifier with **higher a ɑ vlaue** are relevant features!!

→ If window does not contain most relevant features, we can consider the next region on the image.
	 Because the given region does not contain human face



It looks like somewhat decision tree.

<img src="https://user-images.githubusercontent.com/84625523/124781360-4ddaf280-df7e-11eb-9621-8a90d8be4d94.png" alt="Cascade" style="zoom: 33%;" />

Use most relevant feature, and determine whether it's face or not.

* If it isn't move on to next window
* If it is use Second feature.









#### CascadeClassifier.detectMultiScale() parameters

(image, faceDetections, scaleFactor, minNeighbors, flags, minSize, maxSize)

1. scaleFactor: If face is closer to camera, it appears bigger
   * scaleFactor conpensates for this.
   * Specifying how much the image size is reduced at each image scale.
     * The model has a fixed size during training in the haarcascade_frontalface_alt.xml file
     * By rescaling the input image, you can resize a larger face to smaller one
       and make it detectable for algorithm
   * value: 1.1 ~ 1.4
     * small : algorithm will be slow since it is more thorough
     * high : faster detection. but the risck of missing some face altogether.



2. minNeighbors : specifying how many neighbors each candidate rectangle should have to retain it.
   * value: 3 ~ 6
     * higher value : less detections but with higher quality
   * trade off between running time and quality
   * set minNeighbor to 0 -> it will have false positive.
   * Too high minNeighbor : no answer.



3. flags : kind of a heuristic
   * Reject some image region that contain too few or too much edges.
   * Can not contain the searched object



4. minSize: Object smaller than this are ignored
   * We can specify what is the smallest object we want to recognize.
     * [30x30] is standard



5. maxSize : object larger than this is ignored































