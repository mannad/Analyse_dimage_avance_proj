import numpy as np
import skimage.color
import skimage.feature
import skimage.transform

import Bag_of_Words
import utils


def describe_dataset(data, feature='hog', params=None):
    if feature == 'hog':
        result = [hog_job(data, i, params) for i in range(0, len(data))]
        return np.vstack(result)
    elif feature == 'downs_hog':
        # Linearized downsampled 4x4 image concatenated with hog
        f = params['downsample_factor']
        result = []
        for i in range(0, len(data)):
            a = downsample_job(data, i, f)
            b = hog_job(data, i, params)
            combined = np.hstack((a, b))
            result.append(combined)
        return np.vstack(result)
    elif feature == 'hue_hog':
        # Linearized downsampled 4x4 image concatenated with hog
        f = params['downsample_factor']
        result = []
        for i in range(0, len(data)):
            downs_hue = downsample_job(data, i, f)  # downscale and convert to luv
            hogfeat = hog_job(data, i, params)
            combined = np.hstack((downs_hue, hogfeat))  # Keep only the hue
            result.append(combined)
        return np.vstack(result)
    elif feature == 'hog-bow':
        described, centers = Bag_of_Words.create_bags_of_words(data, "hog", params, num_words=params['num_words'])
        return described
    elif feature == 'gray':
        gray_data = skimage.color.rgb2gray(data)
        return utils.flatten_dataset(gray_data)
    elif feature == 'raw':
        return utils.flatten_dataset(data)
    else:
        raise ValueError("Feature is not implemented: " + feature)


def downsample_job(data, i, factor):
    if len(data.shape) == 4:
        downscaled = skimage.transform.downscale_local_mean(data[i], (factor, factor, 1))
        downscaled_with_color_space = skimage.color.rgb2luv(downscaled)
    elif len(data.shape) == 3:
        downscaled_with_color_space = skimage.transform.downscale_local_mean(data[i], (factor, factor))
    else:
        raise Exception("wtf dude")
    return np.reshape(downscaled_with_color_space, (np.prod(downscaled_with_color_space.shape),))


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
