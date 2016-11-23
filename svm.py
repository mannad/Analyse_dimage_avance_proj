import os
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
import pickle

from utils import flatten_dataset
from Bag_of_Words import *

list_datasets = ['mnist', 'cifar10', 'cifar100']
loss = ['hinge', 'squared_hinge']
color_plt = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def loading_data(name_dataset):
    print('Loading data')
    if name_dataset == list_datasets[0]:
        return mnist.load_data()
    elif name_dataset == list_datasets[1]:
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
        plt.text(j, i, '%.2f' % cnf_matrix[i, j], horizontalalignment='center',
                 color='white' if cnf_matrix[i, j] > thresh else 'black')

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
    # plt.xlim(xmin=x_axis[0], xmax=x_axis[-1])  # set axis x within number accuracy
    plt.xlabel('Value of C')
    plt.ylim(0, 100)
    plt.ylabel('Percent of accuracy')
    plt.grid(True)
    # create legend guide
    plt.legend(handles=list_handles)
    plt.savefig('grid_search_svm_' + name_dataset + '.png')
    # plt.show()


def svm(dataset, c=1.0, max_it=1000):
    """

    :param dataset: dataset to train and test
    :type dataset: tuple
    :param c:
    :type c: float
    :param max_it:
    :type max_it: int
    :return: result for loss hinge and squared_hinge
    :rtype: dict{list}
    """
    # Get datasets mnist
    (X_train, y_train), (X_test, y_test) = dataset
    X_train = flatten_dataset(X_train)
    X_test = flatten_dataset(X_test)

    results = {}
    for loss_type in loss:
        # print("#======= %s =======#" % loss_type)
        results[loss_type] = []
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

        results[loss_type].append({'C': c, 'train_accuracy': training_accuracy, 'test_accuracy': test_accuracy})
    return results

# Execute svm
if __name__ == '__main__':
    name_dataset = list_datasets[0]
    print('Loading data %s' % name_dataset)
    (X_train, y_train), (X_test, y_test) = loading_data(name_dataset)

    if os.path.isfile('descr_' + name_dataset + '.bin'):
        print('Loading described data')
        with open('descr_' + name_dataset + '.bin', 'rb') as file:
            (X_train_described, X_test_described) = pickle.load(file)
    else:
        print('Describing training data')
        X_train_described, sift_centers = create_bags_of_words(X_train, debug=True)

        print('Describing test data')
        X_test_described = convert_to_bags(X_test, sift_centers)

        print('Saving described data')
        with open('descr_' + name_dataset + '.bin', 'wb') as file:
            pickle.dump((X_train_described, X_test_described), file, pickle.HIGHEST_PROTOCOL)

    described_data = ((X_train_described, y_train), (X_test_described, y_test))

    list_c = [0.0001, 0.00025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.45, 0.8, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 3]
    res_grid = {loss[0]: [], loss[1]: []}
    for idx_c, c in enumerate(list_c):
        print('Training-Test c=%s' % str(c))
        res = svm(dataset=described_data, c=c)

        res_grid[loss[0]].append(res[loss[0]][0])  # hinge
        res_grid[loss[1]].append(res[loss[1]][0])  # squared_hinge

    # put result in csv
    list_val_name = ['C', 'train_accuracy', 'test_accuracy']
    res_file = open('svm_grid_search_accuracy.txt', 'w')
    for loss_type in res_grid:
        res_file.write('Result %s: \n' % loss_type)
        print('Result %s: \n' % loss_type)
        for r in res_grid[loss_type]:
            print(r)
            line = ''
            for v in list_val_name:
                line += str(r[v]) + ';'
            res_file.write(line + '\n')
        res_file.write('\n')
