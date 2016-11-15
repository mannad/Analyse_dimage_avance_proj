import sys
import itertools
from keras.datasets import mnist, cifar10, cifar100
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
# import cv2.xfeatures2d
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

list_datasets = ['mnist', 'cifar10', 'cifar100']
loss = ['hinge', 'squared_hinge']
color_plt = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def print_progress(it, total):
    """
    Print iterations progress
    :param it: no iteration
    :type it: int
    :param total: number iteration
    :type total: int
    :return: None
    """
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


def plot_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cnf_matrix: confusion matrix to display
    :type cnf_matrix: array, shape = [n_classes, n_classes]
    :param classes: name of classes for label axis x
    :type classes: list
    :param normalize: if we want normalize matrix before to display
    :type normalize: bool
    :param title: title of plot to display
    :type title: str
    :param cmap: color of graduation bar, by default is Blue
    :type cmap: matplotlib.cm
    :return: None
    """
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # print(cm)

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cnf_matrix.max() / 2.0
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, '%.2f' % cnf_matrix[i, j], horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_confusion_matrix(y_test, predicted_y):
    """
    Create confusion matrix and make plot
    :param y_test: result test dataset
    :param predicted_y: result predicted training
    :return: None
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, predicted_y)
    np.set_printoptions(precision=2)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(1, 10), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


def compute_plot_grid_search(a_grid_search, x_axis, name_dataset=list_datasets[0], color_line_plot=None):
    """
    Create plot grid search
    :param a_grid_search: dict category parameter of list each accuracy resulted of test
    :type a_grid_search: dict[list]
    :param color_line_plot: list of color of each category, must be same len of a_grid_search
    :type color_line_plot: list[str]
    :return:
    """
    if not color_line_plot or len(a_grid_search) != len(color_line_plot):
        color_line_plot = color_plt[:len(a_grid_search)]
    # Set params of plot
    list_handles = []
    for idx, type_loss in enumerate(a_grid_search):
        plt.plot(a_grid_search[type_loss], color_line_plot[idx])
        list_handles.append(mpatches.Patch(color=color_line_plot[idx], label=type_loss))
    plt.xlim(xmin=x_axis[0], xmax=x_axis[len(x_axis) - 1])  # set axis x within number accuracy
    plt.xlabel('Value of C')
    plt.ylim(0, 100)
    plt.ylabel('Percent of accuracy')
    plt.grid(True)
    # create legend guide
    plt.legend(handles=list_handles)
    plt.savefig('grid_search_svm_' + name_dataset + '.png')
    plt.show()


def svm(dataset, list_c=[1, 5, 10], max_it=500):
    """

    :param dataset: dataset to train and test
    :type dataset: np.ndarray
    :param list_c:
    :type list_c: list[int]
    :param max_it:
    :type max_it: int
    :return: result for loss hinge and squared_hinge
    :rtype: dict{list}
    """
    nb_it_make = 0
    total_it = 1 + len(loss) * len(list_c)
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
        for c in list_c:
            svm_lin_svc = LinearSVC(C=c, loss=loss_type, max_iter=max_it)

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
            # compute_confusion_matrix(y_test, predicted_y)

            results[loss_type].append({'C': c, 'loss': loss_type,
                                       'diff_accuracy': abs(test_accuracy - training_accuracy),
                                       'accuracy': test_accuracy})

            nb_it_make += 1
            print_progress(nb_it_make, total_it)
    return results

list_c = [1, 10, 30, 60, 100, 500, 1000, 2000]
res = svm(dataset=loading_data(list_datasets[0]), list_c=list_c)
grid_search = {}
grid_search[loss[0]] = [0]
grid_search[loss[1]] = [0]
file = open('result_grid_search_svm' + list_datasets[0] + '.txt', 'w')
for loss_type in res:
    print('Result %s:' % loss_type)
    for r in res[loss_type]:
        print(r)
        file.write(str(r) + '\n')
        grid_search[loss_type].append(r.get('accuracy', 0))

compute_plot_grid_search(grid_search, list_c, list_datasets[0])
