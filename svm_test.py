from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np


def vectorify(a):
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], X_shape[1] * X_shape[2]))


print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# N = 10000
X_train = vectorify(X_train)
y_train = y_train
X_test = vectorify(X_test)
y_test = y_test

print("Training")
svmClassifier = LinearSVC()
svmClassifier.fit(X_train, y_train)

print("Predicting")
predictedY = svmClassifier.predict(X_test)

diff = predictedY - y_test
trainingAccuracy = 100 * (diff == 0).sum() / np.float(len(y_test))
print('Perceptron training accuracy = ', trainingAccuracy, '%')