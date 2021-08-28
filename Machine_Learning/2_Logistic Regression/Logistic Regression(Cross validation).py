import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

credit_data = pd.read_csv("credit_data.csv")

#  setting features, target
features = credit_data[["income", "age", "loan"]]
target = credit_data.default


#  machine learning handle arrays not data-frames
X = np.array(features).reshape(-1, 3)
Y = np.array(target)

model = LogisticRegression()

predicted = cross_validate(model, X, Y, cv=10)
score = cross_val_score(model, X, Y, scoring = 'accuracy', cv=10)


#  cross_validate scores
print(np.mean(predicted['test_score']))
print(np.std(predicted['test_score']))

#  another way to print out scores
print("cross validate scores : {}".format(score))
