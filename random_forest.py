from keras.datasets import mnist, cifar10
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
import numpy as np
import cv2.xfeatures2d


def flatten_dataset(a):
    """Flatten every sample in the dataset
    :param a: ndarray of shape (num_samples , d1 , d2 , ... , dN)
    :return: ndarray of shape  (num_samples , d1 * d2 * ... * dN)
    """
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], np.prod(X_shape[1:])))


def describe_using_sift(data):
    all_features = []  # one item per image, item is of shape (num_keypoints_in_image, 128)
    sample_idx = []  # sample index associated with each keypoint
    sift = cv2.xfeatures2d.SIFT_create()
    count = 0
    for idx, sample in enumerate(data):
        # TODO Seems to work on RGB, does it work as expected?
        key_points, descriptors = sift.detectAndCompute(sample, None)
        if len(key_points) > 0:
            all_features.append(descriptors)
            sample_idx += [idx] * len(key_points)

        count += 1
        if count % 1000 == 0:
            print(count)

    all_features = np.concatenate(all_features)  # shape is (num_keypoints_total, 128)
    sample_idx = np.asarray(sample_idx)

    return all_features, sample_idx


def extract_histograms(all_features, indices, labels, num_samples):
    described_samples = np.zeros((num_samples, 128))
    for i in range(0, len(all_features)):
        described_samples[indices[i], labels[i]] += 1
    linfnorm = np.linalg.norm(described_samples, axis=1, ord=np.inf)  # Get norm of each line
    described_samples.astype(np.float) / linfnorm[:, None]            # Normalize each line
    return described_samples


def make_bags_of_keypoints(data):
    """Make a bags of keypoints representation using SIFT
    :param data: array of samples
    :return: tuple with (1) new representation of each sample (2) list of sift representation centers (words)
    """

    print("Total samples: {}".format(len(data)))

    # 1. Detect SIFT features on all images and add them to one list
    print("Detect SIFT keypoints and compute descriptors")
    all_features, sample_idx = describe_using_sift(data)

    # 2. Do a K-Means (128) on that list of 128-D points
    print("Find descriptors clusters")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    err, labels, centers = cv2.kmeans(data=all_features,
                                      K=128,
                                      bestLabels=None,  # Always None in opencv-python
                                      criteria=criteria,
                                      attempts=3,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    # 3. For each image, make a normalized histogram of those 128 visual words
    print("Make normalized histograms")
    described_samples = extract_histograms(all_features, sample_idx, labels, len(data))

    return described_samples, centers


def convert_to_bags(data, words):
    # 1. compute descriptors of each sample
    all_features, sample_idx = describe_using_sift(data)

    # 2. Insert all words in KDTree
    tree = KDTree(words, leaf_size=2)

    # 3. Find closest word for each sample
    dist, labels = tree.query(all_features)
    described_samples = extract_histograms(all_features, sample_idx, labels, len(data))

    return described_samples


def run_random_forest(data, n_estimators=10, max_features='sqrt', do_predict_training=False):
    ((X_train, y_train),(X_test, y_test)) = data
    y_train = np.reshape(y_train, (y_train.size,))
    y_test = np.reshape(y_test, (y_test.size,))

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

print("Describing training data")
X_train_described, sift_centers = make_bags_of_keypoints(X_train)

print("Describing test data")
X_test_described = convert_to_bags(X_test, sift_centers)

run_random_forest(((X_train_described, y_train),(X_test_described, y_test)), n_estimators=10, do_predict_training=True)
