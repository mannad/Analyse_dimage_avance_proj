from keras.datasets import mnist, cifar10
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cv2.xfeatures2d


def flatten_dataset(a):
    """Flatten every sample in the dataset
    :param a: ndarray of shape (num_samples , d1 , d2 , ... , dN)
    :return: ndarray of shape  (num_samples , d1 * d2 * ... * dN)
    """
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], np.prod(X_shape[1:])))


def make_bags_of_keypoints(data):
    # 1. Detect SIFT features on all images and add them to one list
    all_features = []  # one item per image, item is of shape (num_keypoints_in_image, 128)
    sift = cv2.xfeatures2d.SIFT_create()
    for sample in data:
        # TODO May have to convert to grayscale
        key_points, descriptors = sift.detectAndCompute(sample, None)
        if len(key_points) > 0:
            all_features.append(descriptors)
    all_features = np.concatenate(all_features)  # shape is (num_keypoints_total, 128)

    # 2. Do a K-Means (128) on that list of 32-D points
    # 3. For each image, make a normalized histogram of those 128 visual words
    return None


def run_random_forest(data, n_estimators=10, max_features='sqrt', do_predict_training=False):
    # print("Loading data")
    # (X_train, y_train), (X_test, y_test) = data
    # X_train = flatten_dataset(X_train)
    # y_train = np.reshape(y_train, (y_train.size,))
    # X_test = flatten_dataset(X_test)
    # y_test = np.reshape(y_test, (y_test.size,))

    print("Training")
    rfClassifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1,
                                          verbose=1, random_state=1337)  # n_jobs=-1 => max num cores
    rfClassifier.fit(X_train, y_train)

    if do_predict_training:
        print("Predicting on training")
        predictedY = rfClassifier.predict(X_train)
        diff = predictedY - y_train
        trainingAccuracy = (diff == 0).sum() / np.float(len(y_train))
        print('training accuracy =', trainingAccuracy)
    else:
        print("Not predicting on training, as requested")

    print("Predicting on test dataset")
    predictedY = rfClassifier.predict(X_test)
    diff = predictedY - y_test
    testAccuracy = (diff == 0).sum() / np.float(len(y_test))
    print('test accuracy =', testAccuracy)

    return testAccuracy


print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Describing data")
make_bags_of_keypoints(X_train)

#run_random_forest(mnist.load_data(), n_estimators=10, do_predict_training=True)