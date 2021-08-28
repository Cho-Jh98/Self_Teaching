from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

X_digits = digits.data
y_digits = digits.target
print(X_digits.shape)
# (1797, 64)
estimator = PCA(n_components=2)
# construct covariance matrix and calculate eigenvector
# visualize in 2 axis

X_pca = estimator.fit_transform(X_digits)
print(X_pca.shape)
# (1797, 2)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
# 0~9 as black ~ gray

for i in range(len(colors)):
    px = X_pca[:, 0][y_digits == i]
    py = X_pca[:, 1][y_digits == i]
    plt.scatter(px, py, c=colors[i])  # assign given colors
    plt.legend(digits.target_names)
# scatter plot for dif digit. digit = i >> asign given color based on given index to the given sub data set
# target variable 0 with color[i]

plt.xlabel('first Principle Component')
plt.ylabel('second Principle Component')

plt.show()
# Explained variance shows how much information can be attributed to the principle components.
print("Explained varianc: %s" % estimator.explained_variance_ratio_)



