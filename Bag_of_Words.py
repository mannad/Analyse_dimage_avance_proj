import cv2.xfeatures2d
import numpy as np

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
    # linfnorm = np.linalg.norm(described_samples, axis=1, ord=np.inf)  # Get norm of each line
    linfnorm = np.sum(described_samples, axis=1)
    print("Num empty vectors:", (linfnorm == 0).sum())
    described_samples.astype(np.float) / linfnorm[:, None]  # Normalize each line
    return described_samples
