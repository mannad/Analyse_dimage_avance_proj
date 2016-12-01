import cv2.xfeatures2d
import numpy as np
from sklearn.neighbors import KDTree
from skimage.util.shape import view_as_blocks


def create_bags_of_words(data, num_words=256, debug=False):
    """ Make bags of visual words using SIFT feature
    :param data: array of samples
    :param num_words: the number of words to find
    :param debug: bool to print debug info
    :return: tuple with (1) the new representation of each sample (2) list of sift feature (words)
    """

    print("Total samples: {}".format(len(data)))

    # 1. Detect SIFT features on all images and add them to one list
    print("Detecting SIFT keypoints and computing descriptors...")
    all_features, sample_idx = __describe_using_sift__(data, debug=debug)

    # 2. Do a K-Means (128) on that list of 128-D points
    print("Finding descriptors clusters...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    err, labels, centers = cv2.kmeans(data=all_features,
                                      K=num_words,
                                      bestLabels=None,  # Always None in open-cv-python
                                      criteria=criteria,
                                      attempts=3,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    # 3. For each image, make a normalized histogram of those 128 visual words
    print("Computing normalized histograms...")
    described_samples = __extract_histograms__(all_features, sample_idx, labels, len(data),
                                               num_words=num_words, debug=debug)

    return described_samples, centers


def convert_to_bags(data, words, debug=False):
    """ Convert a sample into a bag of visual words using SIFT feature
    :param data: array of samples
    :param words: array of words to use
    :param debug: bool to print debug info
    :return: array of the new representation of each sample
    """

    # 1. compute descriptors of each sample
    all_features, sample_idx = __describe_using_sift__(data, debug=debug)

    # 2. Insert all words in KDTree
    tree = KDTree(words, leaf_size=2)

    # 3. Find closest word for each sample
    dist, labels = tree.query(all_features)
    described_samples = __extract_histograms__(all_features, sample_idx, labels, len(data),
                                               num_words=len(words), debug=debug)

    return described_samples


def __describe_using_sift__(data, sigma=0.75, debug=False):

    # 1. Convert image to greyscale if needed
    if len(data.shape) == 4:    # RGB images : (nb_samples, channel, rows, cols)
        gray_data = []
        for sample in data:
            out = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
            gray_data.append(out)
        data = gray_data

    # 2. Computing SIFT detection
    all_features = []   # one item per sample, item is of shape (num_keypoints_in_sample, 128)
    sample_idx = []     # sample index associated with each keypoint
    sift = cv2.xfeatures2d.SIFT_create(sigma=sigma)  # see sift_sigma.txt
    for idx, sample in enumerate(data):
        key_points, descriptors = sift.detectAndCompute(sample, None)
        if len(key_points) > 0:
            all_features.append(descriptors)
            sample_idx += [idx] * len(key_points)

    all_features = np.concatenate(all_features)  # shape is (num_keypoints_total, 128)
    sample_idx = np.asarray(sample_idx)

    if debug:
        print("Total features:", len(all_features))

    return all_features, sample_idx


def __describe_using_hog__(data, orientations=8, blocks_per_dim=4):
    assert len(data.shape) == 3  # TODO rgb. For now (n, h, w)

    # Compute gradients for each image
    gradients = np.gradient(data, axis=(1, 2))  # gives (n, h, w, 2)

    # Compute angle of gradients
    grad_orient = np.arctan2(gradients[:, :, :, 0], gradients[:, :, :, 1])  # gives (n, h, w)

    # Split each image into (bpd * bpd) blocks, where bpd = blocks_per_dim
    block_shape = (1, data.shape[1] / blocks_per_dim, data.shape[2] / blocks_per_dim)
    block_size = block_shape[1] * block_shape[2]
    grad_orient_blocks = view_as_blocks(grad_orient, block_shape=block_shape)  # gives (n, bi, bj, bh, bw)

    # Get a matrix where each line is a flattened block (it contains blocks from all images)
    blocks_per_image = blocks_per_dim * blocks_per_dim
    block_list = np.reshape(grad_orient_blocks, (data.shape[0] * blocks_per_image, block_size))

    # Compute histograms, which are our features
    features = []
    indices = np.repeat(np.arange(0, data.shape[0]), blocks_per_image)  # associate feature to image
    histogram_bins = np.linspace(-np.pi, np.pi, orientations + 1)  # ex.: 4 orientations: [-pi, -pi/2, 0, pi/2, pi]
    for block in block_list:
        feat = np.histogram(block, bins=histogram_bins, density=True)  # density=True means that the sum = 1
        features.append(feat)

    return features, indices


def __extract_histograms__(all_features, indices, labels, num_samples, num_words, debug=False):

    # 1. Init histogram
    described_samples = np.zeros((num_samples, num_words))

    # 2. Fill histogram with sample data
    for i in range(0, len(all_features)):
        described_samples[indices[i], labels[i]] += 1

    # 3. Normalize histogram so it's sum is 1
    lin_norm = np.sum(described_samples, axis=1)
    described_samples.astype(np.float) / lin_norm[:, None]

    if debug:
        print("Num empty vectors:", (lin_norm == 0).sum())

    return described_samples
