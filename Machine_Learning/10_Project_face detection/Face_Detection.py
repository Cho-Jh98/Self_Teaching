import cv2
import matplotlib.pyplot as plt

# openCV has a lot of pre-trained classifiers for face detection, eye detection ect.
# several positive and negative samples to train the model (Viola-Jones algorithm)
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# OpenCV deals with BGR but Matplotlib deals with RGB
image = cv2.imread('family_Faces.jpeg')

# this is how we make the transformation
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

# plt.imshow(gray_image, cmap='gray')
# plt.show()

# print(detected_faces)

for(x, y, width, height) in detected_faces:  # top left (x, y) down right (x+w, y+h)

    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 225), 1)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()



