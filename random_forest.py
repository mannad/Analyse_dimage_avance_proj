import cv2.xfeatures2d
import numpy as np
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree

# Number of dimensions of the representation vectors (one vector per image in the dataset)
NUM_BAGS = 256


def describe_using_sift(data):
    if len(data.shape) == 4:  # RGB images : (nb_samples, channel, rows, cols)
        gray_data = []
        for sample in data:
            out = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
            gray_data.append(out)
        data = gray_data

    all_features = []  # one item per image, item is of shape (num_keypoints_in_image, 128)
    sample_idx = []  # sample index associated with each keypoint
    sift = cv2.xfeatures2d.SIFT_create(sigma=0.75)  # see sift_sigma.txt
    for idx, sample in enumerate(data):
        key_points, descriptors = sift.detectAndCompute(sample, None)
        if len(key_points) > 0:
            all_features.append(descriptors)
            sample_idx += [idx] * len(key_points)

    all_features = np.concatenate(all_features)  # shape is (num_keypoints_total, 128)
    sample_idx = np.asarray(sample_idx)
    print("Total features:", len(all_features))

    return all_features, sample_idx


def extract_histograms(all_features, indices, labels, num_samples):
    described_samples = np.zeros((num_samples, NUM_BAGS))
    for i in range(0, len(all_features)):
        described_samples[indices[i], labels[i]] += 1
    linfnorm = np.linalg.norm(described_samples, axis=1, ord=np.inf)  # Get norm of each line
    print("Num empty vectors:", (linfnorm == 0).sum())
    described_samples.astype(np.float) / linfnorm[:, None]  # Normalize each line
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
                                      K=NUM_BAGS,
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

print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

results = []

print("Describing training data")
X_train_described, sift_centers = make_bags_of_keypoints(X_train)

print("Describing test data")
X_test_described = convert_to_bags(X_test, sift_centers)

described_data = ((X_train_described, y_train), (X_test_described, y_test))
accu = run_random_forest(described_data, n_estimators=200, do_predict_training=True)

if len(results) > 0:
    print()
    print("RESULTS:")
    for r in results:
        print(r)
else:
    print("Accuracy:", accu)