import numpy as np
import skimage.feature
from joblib import Parallel, delayed


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
