from keras.datasets import mnist, cifar10
from sklearn.svm import SVC
from features import describe_dataset
import numpy as np
import pickle
import datetime

database = "Cifar10"
feature = "Bow"
gridSearch = False

np.random.seed(159753)

print("Loading data")
if database == "Mnist":
    (X_train_tot, y_train_tot), (X_test, y_test) = mnist.load_data()
    training_length = len(X_train_tot)
    hypParams = [10, 100]
else:
    (X_train_tot, y_train_tot), (X_test, y_test) = cifar10.load_data()
    training_length = len(X_train_tot)
    hypParams = [1, 1]

if feature == "Raw":
    X_train_tot = describe_dataset(X_train_tot, 'raw')
    X_test = describe_dataset(X_test, 'raw')
elif feature == "Gray":
    X_train_tot = describe_dataset(X_train_tot, 'gray')
    X_test = describe_dataset(X_test, 'gray')
elif feature == "Hog":
    X_train_tot = describe_dataset(X_train_tot, 'hog')
    X_test = describe_dataset(X_test, 'hog')
elif feature == "DownHog":
    X_train_tot = describe_dataset(X_train_tot, 'downs_hog',
                                   params={'blocks_per_dim': 4, 'orientations': 6, 'downsample_factor': 8})
    X_test = describe_dataset(X_test, 'downs_hog',
                              params={'blocks_per_dim': 4, 'orientations': 6, 'downsample_factor': 8})
else:
    file = open('descr_bow_cifar10.bin', 'rb')
    (X_train_tot, X_test) = pickle.load(file)
    file.close()

print(database + " " + feature)

if gridSearch:
    # Split training set into training and validation part
    X_train = X_train_tot[0:2000]
    y_train = y_train_tot[0:2000]
    X_valid = X_train_tot[2000:2500]
    y_valid = y_train_tot[2000:2500]

    # Grid search params
    gammaRange = [0.0000001, 0.1, 1, 10, 100]
    slackRange = [0.0000001, 0.1, 1, 10]

    ft = open(database + "_Kernel_" + feature + "_Training.csv", 'w')
    ft.write(';')
    for c in slackRange:
        ft.write(str(c) + ';')
    ft.write('\n')

    fv = open(database + "_Kernel_" + feature + "_VAlidation.csv", 'w')
    fv.write(';')
    for c in slackRange:
        fv.write(str(c) + ';')
    fv.write('\n')

    # grid search
    print('Started at : ' + str(datetime.datetime.now()))
    for gamma_idx, g in enumerate(gammaRange):
        ft.write(str(g) + ';')
        fv.write(str(g) + ';')
        for slack_idx, c in enumerate(slackRange):
            print("Training : \n gamma :" + str(g) + " -|-  c :" + str(c))
            kernelClassifier = SVC(kernel='poly', C=c, gamma=g)
            kernelClassifier.fit(X_train, y_train)

            print("Predicting on training")
            trainingAccuracy = kernelClassifier.score(X_train, y_train)*100
            print('Training accuracy = ', trainingAccuracy, '%')

            print("Predicting on validation")
            validAccuracy = kernelClassifier.score(X_valid, y_valid)*100
            print('Validation accuracy = ', validAccuracy, '%')

            ft.write(str(trainingAccuracy) + ';')
            fv.write(str(validAccuracy) + ';')
        ft.write('\n')
        fv.write('\n')
    ft.write('\n')
    fv.write('\n')
    print('Ended at : ' + str(datetime.datetime.now()))
else:
    # Kernel svm over entire training data set with optimized hyper params
    print("Training with optimal hyper params : \n gamma :" + "%.2f" % hypParams[0] + " -|-  c :" + "%.2f" % hypParams[1])
    kernelClassifier = SVC(kernel='poly', C=hypParams[0], gamma=hypParams[1])
    kernelClassifier.fit(X_train_tot, y_train_tot)

    print("Predicting on training")
    trainingAccuracy = kernelClassifier.score(X_train_tot, y_train_tot)*100
    print('Training accuracy = ', trainingAccuracy, '%')

    print("Predicting on validation")
    validAccuracy = kernelClassifier.score(X_test, y_test)*100
    predictedY = kernelClassifier.predict(X_test)
    print('Validation accuracy = ', validAccuracy, '%')

    f = open(database + "_Kernel_" + feature + "_Result.txt", 'w')
    f.write('training : ' + str('None') + '\nTest : ' + str(validAccuracy) + '\n')
    f.close()

    try:
        print('Compute confusion matrix')
        cnf = np.zeros([10, 10])
        for i, pred in enumerate(predictedY):
            predicted_class = np.argmax(pred)
            real_class = np.argmax(y_test[i])
            cnf[real_class][predicted_class] += 1
        np.savetxt("confusion_matrix" + database + feature + ".csv", cnf)
    except:
        print('Error Matrix')
