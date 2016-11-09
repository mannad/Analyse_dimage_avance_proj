from keras.datasets import mnist
from sklearn.svm import LinearSVC
import numpy as np

def vectorify(a):
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], X_shape[1] * X_shape[2]))

print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = vectorify(X_train)
y_train = y_train
X_test = vectorify(X_test)
y_test = y_test

## BOUT QUI CHANGE ##
print("Training")
svmClassifier = LinearSVC()
svmClassifier.fit(X_train, y_train)
#####################

print("Predicting on training")
predictedY = svmClassifier.predict(X_train)

diff = predictedY - y_train
trainingAccuracy = 100 * (diff == 0).sum() / np.float(len(y_train))
print('training accuracy = ', trainingAccuracy, '%')

print("Predicting on test")
predictedY = svmClassifier.predict(X_test)

diff = predictedY - y_test
trainingAccuracy = 100 * (diff == 0).sum() / np.float(len(y_test))
print('test accuracy = ', trainingAccuracy, '%')