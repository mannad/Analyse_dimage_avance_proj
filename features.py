import numpy as np
import skimage.color
import skimage.feature

import Bag_of_Words


def describe_dataset(data, feature='hog'):
    if feature == 'hog':
        result = [hog_job(data, i) for i in range(0, len(data))]
        return np.vstack(result)
    else:
        raise ValueError("Feature is not implemented: " + feature)


def hog_job(data, i, params=None):
    if params is None:
        params = {'blocks_per_dim': 2, 'orientations': 12}
    b = params['blocks_per_dim']
    img = data[i]
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    return skimage.feature.hog(img,
                               orientations=params['orientations'],
                               pixels_per_cell=(data[0].shape[0] / b, data[0].shape[1] / b),
                               cells_per_block=(1, 1),
                               visualise=False,
                               transform_sqrt=False,
                               feature_vector=True)


def describe_using_bow(train_data, test_data, num_words):
    described_train, words = Bag_of_Words.create_bags_of_words(train_data, num_words=num_words)
    described_test = Bag_of_Words.convert_to_bags(test_data, words)
    return described_train, described_test
