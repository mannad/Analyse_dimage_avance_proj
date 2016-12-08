from keras.datasets import mnist
from sklearn.svm import SVC
import numpy as np
import datetime


def vectorify(a):
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], X_shape[1] * X_shape[2]))

np.random.seed(159753)

print("Loading data")
(X_train_tot, y_train_tot), (X_test, y_test) = mnist.load_data()
training_length = len(X_train_tot)

X_train_tot = vectorify(X_train_tot)
X_test = vectorify(X_test)

# Split training set into training and validation part
X_train = X_train_tot[0:3000]
y_train = y_train_tot[0:3000]
X_valid = X_train_tot[3000:4000]
y_valid = y_train_tot[3000:4000]

# Grid search params
gammaRange = [0.01]
slackRange = [1]

# f = open("KernelGridSearchResult.csv", 'w')
# f.write(';')
# for slack_idx, c in enumerate(slackRange):
#     f.write(str(c) + ';;')
# f.write('\n')

# grid search
print('Started at : ' + str(datetime.datetime.now()))
maxAccuracy = 0.0
hypParams = [0.0, 0.0]
for gamma_idx, g in enumerate(gammaRange):
    # f.write(str(g) + ';')
    for slack_idx, c in enumerate(slackRange):
        f = open("KernelSVM_Gamma_" + str(g) + "_Slack_" + str(c) + ".csv", 'w')

        print("Training : \n gamma :" + "%.5f" % g + " -|-  c :" + "%.2f" % c)
        kernelClassifier = SVC(kernel='poly', C=c, gamma=g)
        kernelClassifier.fit(X_train, y_train)

        print("Predicting on training")
        predictedYTrain = kernelClassifier.predict(X_train)
        diffTrain = predictedYTrain - y_train
        trainingAccuracy = 100 * (diffTrain == 0).sum() / np.float(len(y_train))
        print('Training accuracy = ', trainingAccuracy, '%')

        print("Predicting on validation")
        predictedYValid = kernelClassifier.predict(X_valid)
        diffValid = predictedYValid - y_valid
        validAccuracy = 100 * (diffValid == 0).sum() / np.float(len(y_valid))
        print('Validation accuracy = ', validAccuracy, '%')

        f.write(str(trainingAccuracy) + ';' + str(validAccuracy) + ';')
        f.close()
        # if validAccuracy > maxAccuracy:
        #     maxAccuracy = validAccuracy
        #     hypParams[0] = g
        #     hypParams[1] = c
#     f.write('\n')
# f.write('\n')

# # Kernel svm over entire training data set with optimized hyper params
# print("Training with optimal hyper params : \n gamma :" + "%.2f" % hypParams[0] + " -|-  c :" + "%.2f" % hypParams[1])
# kernelClassifier = SVC()
# kernelClassifier.C = hypParams[0]
# kernelClassifier.gamma = hypParams[1]
# kernelClassifier.fit(X_train[0:len(X_train_tot)], y_train[0:len(X_train_tot)])
#
# print("Predicting on training")
# predictedY = kernelClassifier.predict(X_train_tot)
# diff = predictedY - y_train_tot
# trainingAccuracy = 100 * (diff == 0).sum() / np.float(len(y_train_tot))
# print('Training accuracy = ', trainingAccuracy, '%')
#
# print("Predicting on validation")
# predictedY = kernelClassifier.predict(X_test)
# diff = predictedY - y_test
# testAccuracy = 100 * (diff == 0).sum() / np.float(len(y_test))
# print('Validation accuracy = ', testAccuracy, '%')
#
# f.write(str(trainingAccuracy) + ';' + str(testAccuracy) + ';')
# f.close()
# print('Ended at : ' + str(datetime.datetime.now()))
#
