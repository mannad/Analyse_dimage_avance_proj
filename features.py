import numpy as np
import skimage.feature

import Bag_of_Words


def describe_dataset(data, feature='hog'):
    if feature == 'hog':
        result = [hog_job(data, i) for i in range(0, len(data))]
        return np.vstack(result)
    else:
        raise ValueError("Feature is not implemented: " + feature)


def hog_job(data, i):
    return skimage.feature.hog(data[i],
                               orientations=8,
                               pixels_per_cell=(data[0].shape[0] / 4, data[0].shape[1] / 4),
                               cells_per_block=(1, 1),
                               visualise=False,
                               transform_sqrt=False,
                               feature_vector=True)


def describe_using_bow(train_data, test_data, num_words):
    described_train, words = Bag_of_Words.create_bags_of_words(train_data, num_words=num_words)
    described_test = Bag_of_Words.convert_to_bags(test_data, words)
    return described_train, described_test
