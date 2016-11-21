from keras.datasets import mnist
from sklearn.svm import SVC
import numpy as np


def vectorify(a):
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], X_shape[1] * X_shape[2]))

print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = vectorify(X_train)
X_test = vectorify(X_test)

gammaRange = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
slackRange = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

f = open("KernelGridSearchResult.csv")
for slack_idx, c in enumerate(slackRange):
    f.write(c + ';')
f.write('\n')

for gamma_idx, g in enumerate(gammaRange):
    f.write(g + ';')
    for slack_idx, c in enumerate(slackRange):
        print("Training : g:" + "%.2f" % g + " c:" + "%.2f" % c)
        kernelClassifier = SVC()
        kernelClassifier.C = c
        kernelClassifier.gamma = g
        kernelClassifier.fit(X_train, y_train)

        print("Predicting on training")
        predictedY = kernelClassifier.predict(X_train)

        diff = predictedY - y_train
        trainingAccuracy = 100 * (diff == 0).sum() / np.float(len(y_train))
        print('training accuracy = ', trainingAccuracy, '%')

        print("Predicting on test")
        predictedY = kernelClassifier.predict(X_test)

        diff = predictedY - y_test
        testAccuracy = 100 * (diff == 0).sum() / np.float(len(y_test))
        print('test accuracy = ', testAccuracy, '%')

        f.write(trainingAccuracy + ';' + testAccuracy + ';')
    f.write('\n')
f.close()
