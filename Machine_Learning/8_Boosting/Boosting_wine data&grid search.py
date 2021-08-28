import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing


def is_tasty(quality):
    if quality >= 8:
        return 1
    else:
        return 0


data = pd.read_csv("wine.csv", sep=";")

features = data[
        ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
data['tasty'] = data["quality"].apply(is_tasty)

targets = data['tasty']
# print(features.shape)
# print(target)

# machine learning use array not data-frames
X = np.array(features).reshape(-1, 11)
y = np.array(targets).reshape(-1, 1)
# print(X)

X = preprocessing.MinMaxScaler().fit_transform(X)
# print(X)

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

param_dist = {
    'n_estimators': [10, 50, 200],
    'learning_rate': [0.01, 0.1, 1]
}

estimator = AdaBoostClassifier()

grid_search = GridSearchCV(estimator=estimator, param_grid=param_dist, cv=10)
grid_search.fit(feature_train, target_train)

prediction = grid_search.predict(feature_test)

print(confusion_matrix(target_test, prediction))
print(accuracy_score(target_test, prediction))
