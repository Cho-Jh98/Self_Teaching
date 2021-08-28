import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()
"""
print(digits.images.shape)
#  (1797, 8, 8)
print(digits.target.shape)
#  (1797, )
"""

images_and_label = list(zip(digits.images, digits.target))


"""
#  show image and label
for index, (image, label) in enumerate(images_and_label[:6]):
    plt.subplot(2, 3, index + 1)  # 2 row, 3 column, plot image and label
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')  # gray scale image
    plt.title('Target: %i' % label)
plt.show()
"""

# to apply a classifier on this data, we need to flatten the image : instead of a 8x8 matirx,
# we have to use a one-dimensional array with 64 items
data = digits.images.reshape((len(digits.images), -1)) #  reshape(# of row, # of col)

"""
print(data.shape)
# 실행결과 : (1797, 64)
"""
param_grid = { 'C' : [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}

classifier = svm.SVC()

#  70% of original data-set is for training
train_test_split = int(len(digits.images) * 0.7)
grid = GridSearchCV(classifier, param_grid, refit=True)

grid.fit(data[:train_test_split], digits.target[:train_test_split])
print(grid.best_estimator_)

grid_prediction = grid.predict(data[train_test_split:])
print(confusion_matrix(digits.target[train_test_split:], grid_prediction))
print(accuracy_score(digits.target[train_test_split:], grid_prediction))
#  classifier.fit(data[:train_test_split], digits.target[:train_test_split])


"""
#  now predict the value of the digit on the 25%
expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

print("confusion matrix: \n%s" % metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))

# let's test on the last few images
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
print("prediction for the test image: ", classifier.predict(data[-1].reshape(1,-1)))

plt.show()
"""