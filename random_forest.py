import numpy as np
import os
import pickle

import time
from keras.datasets import mnist, cifar10
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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

    np.savetxt("predict_rf.csv", predictedY)
    np.savetxt("accu.csv", np.array((trainingAccuracy, testAccuracy)))

    confmat = confusion_matrix(y_test, predictedY)
    np.savetxt("confusion_rf.csv", confmat)

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
X_train_all = np.vstack((X_train_all, X_test))
y_train_all = np.concatenate((y_train_all, y_test))
N_TRAIN = 50000  # For this run, The training set is all training samples and the validation set is the test set.
X_train = X_train_all[:N_TRAIN]
y_train = y_train_all[:N_TRAIN]
X_valid = X_train_all[N_TRAIN:]
y_valid = y_train_all[N_TRAIN:]

feature_type = "gray"
feature_params = {}  # 'num_words': 128, 'orientations': 8, 'blocks_per_dim': 4}

if False:  # os.path.isfile("descr.bin"):
    print("Loading described data")
    with open("descr.bin", "rb") as file:
        (X_train_described, X_valid_described) = pickle.load(file)
else:
    print("Describing training data")
    X_train_all_described = features.describe_dataset(X_train_all, feature=feature_type, params=feature_params)
    X_train_described = X_train_all_described[:N_TRAIN]
    assert X_train_described.shape[0] == len(X_train)

    print("Describing test data")
    X_valid_described = X_train_all_described[N_TRAIN:]

    print("Saving described data")
    with open("descr.bin", "wb") as file:
        pickle.dump((X_train_described, X_valid_described), file, pickle.HIGHEST_PROTOCOL)

described_data = ((X_train_described, y_train), (X_valid_described, y_valid))

# described_data = ((utils.flatten_dataset(X_train), y_train), (utils.flatten_dataset(X_valid), y_valid))

# Perform grid search
min_s_split_values = [4]
max_features_values = [128]
results = np.zeros((len(min_s_split_values), len(max_features_values), 3))

with open("cumulative_results.txt", "a") as file:
    file.write("\n\nNew grid search ==== " + time.ctime() + "\n")
    file.write("Feature: {}  params: {}\n".format(feature_type, feature_params))

for i, min_s_split in enumerate(min_s_split_values):
    for j, max_features in enumerate(max_features_values):
        accu = run_random_forest(described_data, n_estimators=1000, max_features=max_features,
                                 min_samples_split=min_s_split, do_predict_training=True)
        results[i, j] = accu
        with open("cumulative_results.txt", "a") as file:
            file.write("{:<4} {:<2} {:<.3f} {:<.3f} {:<.3f}\n".format(min_s_split, max_features, accu[0], accu[2], accu[1]))

with open("results.bin", "wb") as file:
    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

np.savetxt("gridsearch_accuracy_train.csv", results[:, :, 0], delimiter="\t")
np.savetxt("gridsearch_accuracy_test.csv", results[:, :, 1], delimiter="\t")
np.savetxt("gridsearch_accuracy_oob.csv", results[:, :, 2], delimiter="\t")
print("See grid search results in csv files")
