import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

data = pd.read_csv("credit_data.csv")

features = data[["income", "age", "loan"]]
target = data.default

X = np.array(features).reshape(-1, 3)
y = np.array(target)


X = preprocessing.MinMaxScaler().fit_transform(X)

feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size= 0.3)

model = KNeighborsClassifier(n_neighbors=32)
fitted_model = model.fit(feature_train, target_train)
prediction = fitted_model.predict(feature_test)

cross_validation_scores = []

for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())

print("optimal k is : ", np.argmax(cross_validation_scores))


print(confusion_matrix(target_test, prediction))
print(accuracy_score(target_test, prediction))
