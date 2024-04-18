import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from ipywidgets import interact
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

"""
K Nearest Neighbors

KNN is a classification algorithm takes input some data and use the data to determine which class (category) this piece 
of data belongs to.

The iris dataset consists of 150 data points where each data point is a feature vector x consisting of 4 features 
describing the attribute of a flower in the dataset, the four dimensions represent:
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 

And the corresponding target y in Z describes the class of the flower. It uses the integers 0, 1 and 2 to represent 
the 3 classes of flowers in this dataset.
0. Iris Setosa
1. Iris Versicolour 
2. Iris Virginica
"""

iris = datasets.load_iris()
print('data shape is {}'.format(iris.data.shape))
print('class shape is {}'.format(iris.target.shape))

# use first two version for simplicity
X = iris.data[:, :2]
y = iris.target

iris = datasets.load_iris()
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

fig, ax = plt.subplots(figsize=(4, 4))
for i, iris_class in enumerate(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']):
    idx = y == i
    ax.scatter(X[idx, 0], X[idx, 1],
               c=cmap_bold.colors[i], edgecolor='k',
               s=20, label=iris_class);
ax.set(xlabel='sepal length (cm)', ylabel='sepal width (cm)')
ax.legend()
plt.show()


def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)

    Returns
    --------
    D: matrix of shape (N, M), each entry D[i,j] is the distance between
    X[i] and Y[j] using the dot product.
    """
    N, D = X.shape
    M, _ = Y.shape
    distance_matrix = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            x_min_y = X[n] - Y[m]
            distance_matrix[n][m] = np.sqrt(x_min_y.T @ x_min_y)
    return distance_matrix


def KNN(k, X, y, x):
    """
    K nearest neighbors

    k: number of nearest neighbors
    X: training input locations
    y: training labels
    x: test input
    """
    num_classes = len(np.unique(y))
    dist = pairwise_distance_matrix(X, np.array([x]))
    dist = np.reshape(dist, newshape=(1, -1))[0]

    # Next we make the predictions
    ypred = np.zeros(num_classes)
    classes = y[np.argsort(dist)][:k]  # find the labels of the k nearest neighbors
    for c in np.unique(classes):
        ypred[c] += 1

    return np.argmax(ypred)


"""
We can also visualize the "decision boundary" of the KNN classifier, which is the region of a problem space in which 
the output label of a classifier is ambiguous. This would help us develop an intuition of how KNN behaves in practice. 
The code below plots the decision boundary.
"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

K = 3
ypred = []
for xtest in np.array([xx.ravel(), yy.ravel()]).T:
    ypred.append(KNN(K, X, y, xtest))

fig, ax = plt.subplots(figsize=(4, 4))

ax.pcolormesh(xx, yy, np.array(ypred).reshape(xx.shape), cmap=cmap_light)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.show()
