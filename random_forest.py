import numpy as np
import os
import pickle
from keras.datasets import mnist, cifar10
from sklearn.ensemble import RandomForestClassifier
import features


def run_random_forest(data, n_estimators, max_features, do_predict_training=False):
    ((X_train, y_train), (X_test, y_test)) = data
    y_train = np.reshape(y_train, (y_train.size,))
    y_test = np.reshape(y_test, (y_test.size,))

    print("Training")
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1,
                                           verbose=1, random_state=1337)  # n_jobs=-1 => max num cores
    rf_classifier.fit(X_train, y_train)

    trainingAccuracy = 0
    if do_predict_training:
        print("Predicting on training")
        predictedY = rf_classifier.predict(X_train)
        diff = predictedY - y_train
        trainingAccuracy = (diff == 0).sum() / np.float(len(y_train))
        print('training accuracy =', trainingAccuracy)
    else:
        print("Not predicting on training, as requested")

    print("Predicting on test dataset")
    predictedY = rf_classifier.predict(X_test)
    diff = predictedY - y_test
    testAccuracy = (diff == 0).sum() / np.float(len(y_test))
    print('test accuracy =', testAccuracy)

    return trainingAccuracy, testAccuracy


# TODO À tester
#   - Description des images test
#   - x Sift trop puissant pour taille des images? (overfitting)
#   - Assez de keypoints, assez de words par image?
#   - Threshold SIFT

# Best MNIST-SIFT to date: 84%
# Best MNIST-Raw  to date: 97%

print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if os.path.isfile("descr.bin"):
    print("Loading described data")
    with open("descr.bin", "rb") as file:
        (X_train_described, X_test_described) = pickle.load(file)
else:
    print("Describing training data")
    X_train_described = features.describe_dataset(X_train, feature="hog")
    assert X_train_described.shape[0] == len(X_train)

    print("Describing test data")
    X_test_described = features.describe_dataset(X_test, feature="hog")

    print("Saving described data")
    with open("descr.bin", "wb") as file:
        pickle.dump((X_train_described, X_test_described), file, pickle.HIGHEST_PROTOCOL)

described_data = ((X_train_described, y_train), (X_test_described, y_test))

# Perform grid search
num_est_values = [200]  # [50, 100, 200, 1000]
max_features_values = [0.01]  # [2, 4, 8, 16, 32]
results = np.zeros((len(num_est_values), len(max_features_values), 2))

for i, num_est in enumerate(num_est_values):
    for j, max_features in enumerate(max_features_values):
        accu = run_random_forest(described_data, n_estimators=num_est, max_features=max_features,
                                 do_predict_training=True)
        results[i, j] = accu

with open("results.bin", "wb") as file:
    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

np.savetxt("gridsearch_accuracy_train.csv", results[:, :, 0], delimiter=";")
np.savetxt("gridsearch_accuracy_test.csv", results[:, :, 1], delimiter=";")
print("See grid search results in csv files")
