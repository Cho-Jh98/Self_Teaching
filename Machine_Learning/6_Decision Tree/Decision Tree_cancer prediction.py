import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

cancer_data = datasets.load_breast_cancer()

features = cancer_data.data
labels = cancer_data.target

print(features.shape)

feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.2)

model = DecisionTreeClassifier(criterion='entropy', max_depth=8)

predicted = cross_validate(model, features, labels, cv=10)
print(np.mean(predicted['test_score']))
