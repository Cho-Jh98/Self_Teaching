from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)
# estimator : number of used weak learner
# learning_rate : trade-off with estimator
# random_state : random seed for base estimator
# algorithm : SAMME, SAMME.R >> 1 or 2 additional assumption, make converge faster
model.fitted = model.fit(feature_train, target_train)
model.prediction = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.prediction))
print(accuracy_score(target_test, model.prediction))
