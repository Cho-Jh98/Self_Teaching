from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score


# load data-set
mnist_data = fetch_openml('mnist_784')

features = mnist_data.data
targets = mnist_data.target
#  60,000 training sample, 10,000 test sample.
#  target = 28 x 28
#  print(features.shape)
#  (70000, 784)

train_img, test_img, train_lbl, test_lbl = train_test_split(features, targets, test_size=0.15, random_state=123)

scaler = StandardScaler()
scaler.fit(train_img)  # calculate the mean and S for later scaling
train_img = scaler.transform(train_img)  # perform standardization by centering and scaling
test_img = scaler.transform(test_img)

#  print(train_img.shape)
#  (70000, 784) << original dataset so no change

# We keep 95% of variance >> so 95% of the original information is used
#
pca = PCA(.95)
pca.fit(train_img)

# Transform
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


#  print(train_img.shape)
#  (70000, 328) << approximately 400 features are removed while keeping 95% of original information
#  instead of fitting 784 feature we can use 328 feature(almost half of feature)

#  but DL can use original feature as whole >> much powerful
