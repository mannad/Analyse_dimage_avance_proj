import sys
import itertools
from keras.datasets import mnist, cifar10, cifar100
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
# import cv2.xfeatures2d
import matplotlib.pyplot as plt

dict_datasets = ['mnist', 'cifar10', 'cifar100']
loss = ['hinge', 'squared_hinge']


# Print iterations progress
def print_progress(it, total):
    percent = 100 * (it / total)
    sys.stdout.write('Progress: %.0f %% ' % percent)
    if it == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def set_datasets_in_1D(a_datasets):
    """
    Flatten dataset like this (samples, feature) matrix
    :param a_datasets: datasets loading
    :type a_datasets: numpy.ndarray
    :return: new shape
    :rtype: numpy.ndarray
    """
    X_shape = a_datasets.shape
    return np.reshape(a_datasets, (X_shape[0], X_shape[1] * X_shape[2]))


def loading_data(name_dataset):
    print("Loading data")
    if name_dataset == dict_datasets[0]:
        return mnist.load_data()
    elif name_dataset == dict_datasets[1]:
        return cifar10.load_data()
    else:
        return cifar100.load_data()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f' % cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_confusion_matrix(y_test, predicted_y):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, predicted_y)
    np.set_printoptions(precision=2)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(1, 10), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


def svm(dataset, c_min=1.0, c_max=10.0, step_c=1, min_it=100, max_it=1000, step_it=100):
    nb_it_make = 0
    total_it = 1 + len(loss) * \
                   len(np.arange(c_min, c_max + step_c, step_c)) * \
                   len(range(min_it, max_it + step_it, step_it))
    print_progress(nb_it_make, total_it)
    # Get datasets mnist
    (X_train, y_train), (X_test, y_test) = dataset
    X_train = set_datasets_in_1D(X_train)
    X_test = set_datasets_in_1D(X_test)

    nb_it_make += 1
    print_progress(nb_it_make, total_it)

    results = {}
    for loss_type in loss:
        # print("#======= %s =======#" % loss_type)
        results[loss_type] = []
        for c in np.arange(c_min, c_max + step_c, step_c):
            for nb_it in range(min_it, max_it + step_it, step_it):
                svm_lin_svc = LinearSVC(C=c, loss=loss_type, max_iter=nb_it)

                # print("======= Training =======")
                svm_lin_svc.fit(X_train, y_train)
                # print("Predicting on training")
                predicted_y = svm_lin_svc.predict(X_train)
                diff = predicted_y - y_train
                training_accuracy = 100 * (diff == 0).sum() / np.float(len(y_train))
                # print('Training accuracy = ', '%.2f' % training_accuracy, '%')
                # print('Scores', svm_lin_svc.score(X_train, y_train))

                # print("======= Test =======")
                # print("Predicting on test")
                predicted_y = svm_lin_svc.predict(X_test)
                diff = predicted_y - y_test
                test_accuracy = 100 * (diff == 0).sum() / np.float(len(y_test))
                # print('Test accuracy = ', '%.2f' % test_accuracy, '%')
                # print('Scores', svm_lin_svc.score(X_test, y_test))

                # Compute confusion matrix
                compute_confusion_matrix(y_test, predicted_y)

                results[loss_type].append({'nb_it': nb_it, 'c': c, 'loss': loss_type,
                                           'training_accuracy': training_accuracy, 'test_accuracy': test_accuracy})

                nb_it_make += 1
                print_progress(nb_it_make, total_it)
    return results

res = svm(dataset=loading_data(dict_datasets[0]), c_min=1.0, c_max=1.0, min_it=100, max_it=100)
for loss in res:
    print('Result %s:' % loss)
    for r in res[loss]:
        print(r)
