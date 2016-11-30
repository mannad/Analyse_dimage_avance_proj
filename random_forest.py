import numpy as np
import os
import pickle

import time
from keras.datasets import mnist, cifar10
from sklearn.ensemble import RandomForestClassifier
import features
import utils


def run_random_forest(data, n_estimators, max_features, min_samples_split, do_predict_training=False):
    ((X_train, y_train), (X_test, y_test)) = data
    y_train = np.reshape(y_train, (y_train.size,))
    y_test = np.reshape(y_test, (y_test.size,))

    print("Training")
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1,
                                           verbose=1, random_state=1337, oob_score=True,
                                           min_samples_split=min_samples_split)  # n_jobs=-1 => max num cores
    rf_classifier.fit(X_train, y_train)
    fit_oob_score = rf_classifier.oob_score_

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

    return trainingAccuracy, testAccuracy, fit_oob_score


# TODO À tester
#   - Description des images test
#   - x Sift trop puissant pour taille des images? (overfitting)
#   - Assez de keypoints, assez de words par image?
#   - Threshold SIFT

# Best MNIST-SIFT to date: 84%
# Best MNIST-Raw  to date: 97%

print("Loading data")
(X_train_all, y_train_all), (X_test, y_test) = cifar10.load_data()
X_train = X_train_all[0:40000]
y_train = y_train_all[0:40000]
X_valid = X_train_all[40000:50000]
y_valid = y_train_all[40000:50000]

feature_type = "downs_hog"
feature_params = {'blocks_per_dim': 4, 'orientations': 6, 'downsample_factor': 8}

if os.path.isfile("descr.bin"):
    print("Loading described data")
    with open("descr.bin", "rb") as file:
        (X_train_described, X_valid_described) = pickle.load(file)
else:
    print("Describing training data")
    X_train_described = features.describe_dataset(X_train, feature=feature_type, params=feature_params)
    assert X_train_described.shape[0] == len(X_train)

    print("Describing test data")
    X_valid_described = features.describe_dataset(X_valid, feature=feature_type, params=feature_params)

    print("Saving described data")
    with open("descr.bin", "wb") as file:
        pickle.dump((X_train_described, X_valid_described), file, pickle.HIGHEST_PROTOCOL)

assert X_train_described.shape[1] == 144

described_data = ((X_train_described, y_train), (X_valid_described, y_valid))

# Perform grid search
# num_est_values = [500, 1000, 2000]  # [50, 100, 200, 1000]
min_s_split_values = [16]
max_features_values = [16]
results = np.zeros((len(min_s_split_values), len(max_features_values), 3))

with open("cumulative_results.txt", "a") as file:
    file.write("\n\nNew grid search ==== " + time.ctime() + "\n")
    file.write("Hog params: " + str(feature_params) + "\n")

#for i, num_est in enumerate(num_est_values):
for i, min_s_split in enumerate(min_s_split_values):
    for j, max_features in enumerate(max_features_values):
        accu = run_random_forest(described_data, n_estimators=500, max_features=max_features,
                                 min_samples_split=min_s_split, do_predict_training=True)
        results[i, j] = accu
        with open("cumulative_results.txt", "a") as file:
            file.write("{:<4} {:<2} {:<.3f} {:<.3f} {:<.3f}\n".format(min_s_split, max_features, accu[0], accu[2], accu[1]))

with open("results.bin", "wb") as file:
    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

np.savetxt("gridsearch_accuracy_train.csv", results[:, :, 0], delimiter=",")
np.savetxt("gridsearch_accuracy_test.csv", results[:, :, 1], delimiter=",")
print("See grid search results in csv files")
