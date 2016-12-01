import os
import itertools
from keras.datasets import mnist, cifar10, cifar100
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle

from utils import flatten_dataset
from features import describe_dataset, describe_using_bow

list_datasets = ['mnist', 'cifar10', 'cifar100']
list_loss = ['hinge', 'squared_hinge']
list_feature = ['raw_pixel', 'bow', 'hog']
color_plt = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
list_class_label_tag = {list_datasets[0]: list(range(0, 10)),
                        list_datasets[1]: ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                           'dog', 'frog', 'horse', 'ship', 'truck'],
                        list_datasets[2]: ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish', 'flatfish',
                                           'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses', 'sunflowers',
                                           'tulips', 'bottles','bowls', 'cans', 'cups', 'plates', 'apples', 'mushrooms',
                                           'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp',
                                           'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe',
                                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard',
                                           'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper',
                                           'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle',
                                           'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum',
                                           'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby',
                                           'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake',
                                           'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak',
                                           'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup truck',
                                           'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']}


def loading_data(name_dataset):
    """
    Loading data
    :param name_dataset: name data to extract
    :type name_dataset: str
    :return: data
    :rtype: np.ndarray
    """
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
    :type classes: list | range
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


def compute_confusion_matrix(y_test, predicted_y, type_feature, show_plot=False):
    """
    Create confusion matrix and make plot to show (optional)
    :param y_test: result test dataset
    :type y_test: np.ndarray
    :param predicted_y: result predicted training
    :type predicted_y: np.ndarray
    :return: matrix of confusion
    :rtype: np.ndarray
    """
    print('Compute confusion matrix')
    cnf_matrix = confusion_matrix(y_test, predicted_y)
    # Plot normalized confusion matrix
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list_class_label_tag[type_feature], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('cnf_matrix_svm_' + type_feature + '.png')
    if show_plot:
        plt.show()
    else:
        plt.close()
    return cnf_matrix


def compute_plot_grid_search(a_grid_search, x_axis, name_dataset=list_datasets[0], color_line_plot=None):
    """
    Create plot grid search
    :param a_grid_search: dict category parameter of list each accuracy resulted of test
    :type a_grid_search: dict[list]
    :param color_line_plot: list of color of each category, must be same len of a_grid_search
    :type color_line_plot: list[str]
    :return: None
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
    # X_train = flatten_dataset(X_train).astype('float32') / 255
    # X_test = flatten_dataset(X_test).astype('float32') / 255
    X_train = flatten_dataset(X_train)
    X_test = flatten_dataset(X_test)

    results = {}
    for loss_type in list_loss:
        results[loss_type] = []
        svm_lin_svc = LinearSVC(C=c, loss=loss_type, max_iter=max_it)

        # Training
        svm_lin_svc.fit(X_train, y_train)
        predicted_y = svm_lin_svc.predict(X_train)
        diff = predicted_y - y_train
        training_accuracy = 100 * (diff == 0).sum() / np.float(len(y_train))
        # Test
        predicted_y = svm_lin_svc.predict(X_test)
        diff = predicted_y - y_test
        test_accuracy = 100 * (diff == 0).sum() / np.float(len(y_test))

        results[loss_type].append({'C': c, 'train_accuracy': training_accuracy, 'test_accuracy': test_accuracy})
    return results


def processing_data(X_train, X_test, type_feature):
    """
    Processing data depending of the type feature
    :param X_train: data for training
    :type X_train: array
    :param X_test: data for test
    :type X_test: array
    :param type_feature: name of type feature
    :type type_feature: str
    :return: X_train and X_test data feature or same data and list of c depending type feature
    :rtype: array, array, list
    """
    if type_feature == list_feature[0]:
        X_train_described = X_train
        X_test_described = X_test
        list_c = [0.0000001, 0.000001, 0.00001, 0.0001, 0.00025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.45,
                  0.8, 1.0, 1.25, 1.5, 1.75, 2]
    else:
        file_bin = 'descr_' + type_feature + '_' + name_dataset + '.bin'
        if os.path.isfile(file_bin):
            print('Loading described data')
            with open(file_bin, 'rb') as file:
                (X_train_described, X_test_described) = pickle.load(file)
        elif type_feature == list_feature[1]:  # bow
            X_train_described, X_test_described = describe_using_bow(X_train, X_test, 256)
            print('Saving described data')
            with open(file_bin, 'wb') as file:
                pickle.dump((X_train_described, X_test_described), file, pickle.HIGHEST_PROTOCOL)
        elif type_feature == list_feature[2]:  # hog
            print('feature hog')
            X_train_described = describe_dataset(X_train, type_feature)
            X_test_described = describe_dataset(X_test, type_feature)
        else:
            raise ValueError("Feature is not implemented: " + type_feature)

        if type_feature == list_feature[1]:  # bow
            list_c = [0.0001, 0.00025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.45, 0.8, 1.0, 1.25, 1.5, 1.75,
                      2, 2.5, 3, 4, 5]
        else:  # hog
            list_c = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15]
    return X_train_described, X_test_described, list_c

# Execute svm
list_type_feature = [list_feature[2]]
if __name__ == '__main__':
    res_compute_svm_file = open('svm_result.txt', 'w')
    for name_dataset in list_datasets[:1]:
        print('======= Loading data %s =======' % name_dataset)
        (X_train, y_train), (X_test, y_test) = loading_data(name_dataset)
        res_compute_svm_file.write('============== %s ==============' % name_dataset)

        # X_train_train = X_train[:int(len(X_train)*0.7)]
        # X_train_valid = X_train[int(len(X_train)*0.7):]
        # y_train_train = y_train[:int(len(y_train) * 0.7)]
        # y_train_valid = y_train[int(len(y_train) * 0.7):]
        for type_feature in list_type_feature:
            print('=== Feature %s ===' % type_feature)
            print('=== Training ===')
            # X_train_described, X_test_described, list_c = processing_data(X_train_train, X_train_valid, type_feature)
            # described_data = ((X_train_described, y_train_train), (X_test_described, y_train_valid))
            X_train_described, X_test_described, list_c = processing_data(X_train, X_test, type_feature)
            described_data = ((X_train_described, y_train), (X_test_described, y_test))

            res_grid = {list_loss[0]: [], list_loss[1]: []}
            best_hyper_params = {'C': 0.0, 'accuracy': 0.0, 'loss': None}
            for idx_c, c in enumerate(list_c):
                print('Training c=%s' % str(c))
                res = svm(dataset=described_data, c=c)

                for loss_type in list_loss:
                    res_grid[loss_type].append(res[loss_type][0])
                    if (best_hyper_params['accuracy'] < res[loss_type][0]['test_accuracy']) \
                            and abs(res[loss_type][0]['train_accuracy'] - res[loss_type][0]['test_accuracy']) < 1.0:
                        best_hyper_params['C'] = c
                        best_hyper_params['accuracy'] = res[loss_type][0]['test_accuracy']
                        best_hyper_params['loss'] = loss_type

            # put result in csv
            list_val_name = ['C', 'train_accuracy', 'test_accuracy']
            res_file = open('svm_grid_search_accuracy_' + type_feature + '_' + name_dataset + '.txt', 'w')
            for loss_type in res_grid:
                res_file.write('Result %s;Train;Test;Diff;\n' % loss_type)
                print('Result %s: \n' % loss_type)
                for r in res_grid[loss_type]:
                    print(r)
                    line = ''
                    for v in list_val_name:
                        line += str(r[v]) + ';'
                    line += str(r[list_val_name[1]] - r[list_val_name[2]]) + ';'
                    res_file.write(line + '\n')
                res_file.write('\n')

            print('=== Test c=%s accuracy_training=%s loss=%s ==='
                  % (str(best_hyper_params['C']), str(best_hyper_params['accuracy']), str(best_hyper_params['loss'])))

            # svm_lin_svc = LinearSVC(C=best_hyper_params['C'], loss=best_hyper_params['loss'])
            # predicted_y = svm_lin_svc.predict(X_test)
            # diff = predicted_y - y_test
            # training_accuracy = 100 * (diff == 0).sum() / np.float(len(y_test))
            #
            # # Compute confusion matrix
            # cnf_matrix = compute_confusion_matrix(y_test, predicted_y, type_feature)
            # res_compute_svm_file.write('c=%s;accuracy_training=%s;loss=%s;matrix_confusion /n'
            #                             % (str(best_hyper_params['C']), str(best_hyper_params['accuracy']),
            #                                str(best_hyper_params['loss'])))
            # res_compute_svm_file.write(str(cnf_matrix))
            # res_compute_svm_file.write('\n')
    res_compute_svm_file.close()
