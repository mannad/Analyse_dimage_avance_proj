import numpy as np

def flatten_dataset(a):
    """Flatten every sample in the dataset
    :param a: ndarray of shape (num_samples , d1 , d2 , ... , dN)
    :return: ndarray of shape  (num_samples , d1 * d2 * ... * dN)
    """
    X_shape = a.shape
    return np.reshape(a, (X_shape[0], np.prod(X_shape[1:])))
