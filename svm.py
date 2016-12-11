import os
import itertools
from keras.datasets import mnist, cifar10
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle

from utils import flatten_dataset
from features import describe_dataset, describe_using_bow

list_datasets = ['mnist', 'cifar10']
list_loss = ['hinge', 'squared_hinge']
list_feature = ['raw_pixel', 'bow', 'hog', 'gray', 'downs_hog']
color_plt = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
list_class_label_tag = {list_datasets[0]: list(range(0, 10)),
                        list_datasets[1]: ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                           'dog', 'frog', 'horse', 'ship', 'truck']}


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
        raise ValueError('Not exists in list datasets %s' % name_datasets)


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


def compute_confusion_matrix(y_test, predicted_y, name_dataset, type_feature, show_plot=False):
    """
    Create confusion matrix and make plot to show (optional)
    :param y_test: result test dataset
    :type y_test: np.ndarray
    :param predicted_y: result predicted training
    :type predicted_y: np.ndarray
    :param name_dataset: name data to extract
    :type name_dataset: str
    :param type_feature: name of type feature
    :type type_feature: str
    :param show_plot: if we want show plot cnf matrix, put to True
    :type show_plot: bool
    :return: matrix of confusion
    :rtype: np.ndarray
    """
    print('Compute confusion matrix')
    cnf_matrix = confusion_matrix(y_test, predicted_y)
    # Plot normalized confusion matrix
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list_class_label_tag[name_dataset], normalize=True,
                          title='Normalized confusion matrix %s %s' % (name_dataset, type_feature))
    plt.savefig('cnf_matrix_svm_' + name_dataset + '_' + type_feature + '.png')
    if show_plot:
        plt.show()
    else:
        plt.close()
    return cnf_matrix


def compute_plot_grid_search(a_grid_search, name_datasets, type_feature, color_line_plot=None):
    """
    Create plot grid search
    :param a_grid_search: dict category parameter of list each accuracy resulted of test
    :type a_grid_search: dict[list]
    :param name_datasets: name data to extract
    :type name_datasets: str
    :param type_feature: name of type feature
    :type type_feature: str
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

    plt.xlabel('Value of C')
    plt.ylim(0, 100)
    plt.ylabel('Percent of accuracy')
    plt.grid(True)
    # create legend guide
    plt.legend(handles=list_handles)
    plt.savefig('grid_search_svm_' + name_datasets + '_' + type_feature + '.png')
    # plt.show()


def svm(dataset, c=1.0, max_it=1000, loss=None):
    """
    Run svm
    :param dataset: dataset to train and test
    :type dataset: tuple
    :param c:
    :type c: float
    :param max_it:
    :type max_it: int
    :param loss: loss specific
    :type loss: str
    :return: result for loss hinge and squared_hinge
    :rtype: dict{list}
    """
    if loss is None:
        loss = list_loss
    else:  # put type loss in list
        loss = [loss]

    (X_train, y_train), (X_test, y_test) = dataset
    results = {}
    for loss_type in loss:
        results[loss_type] = []
        svm_lin_svc = LinearSVC(C=c, loss=loss_type, max_iter=max_it, random_state=1337)

        # Training
        svm_lin_svc.fit(X_train, y_train)
        training_accuracy = svm_lin_svc.score(X_train, y_train) * 100
        # Test
        predicted_y = svm_lin_svc.predict(X_test)
        test_accuracy = svm_lin_svc.score(X_test, y_test) * 100

        results[loss_type].append({'C': c, 'train_accuracy': training_accuracy, 'test_accuracy': test_accuracy,
                                   'predicted_y': predicted_y})
    return results


def processing_data(X_train, X_test, type_feature, name_datasets, is_train=True):
    """
    Processing data depending of the type feature
    :param X_train: data for training
    :type X_train: array
    :param X_test: data for test
    :type X_test: array
    :param type_feature: name of type feature
    :type type_feature: str
    :param name_datasets: name data to extract
    :type name_datasets: str
    :param is_train: if training, return list_c
    :type is_train: bool
    :return: X_train and X_test data feature or same data and list of c depending type feature
    :rtype: array, array, list
    """
    dict_call_feature = {'raw_pixel': 'raw', 'bow': 'hog-bow', 'hog': 'hog', 'gray': 'gray', 'downs_hog': 'downs_hog'}
    # get name file
    if is_train:
        file_bin = 'descr_' + type_feature + '_' + name_datasets + '_train.bin'
    else:
        file_bin = 'descr_' + type_feature + '_' + name_datasets + '.bin'
    # get data
    if os.path.isfile(file_bin):
        print('Loading described data')
        file = open(file_bin, 'rb')
        (X_train_described, X_test_described) = pickle.load(file)
        file.close()
    else:
        if name_datasets == list_datasets[0]:
            feature_params = {'blocks_per_dim': 4, 'orientations': 6, 'downsample_factor': 4, 'num_words': 128}
        elif name_datasets == list_datasets[1]:
            feature_params = {'blocks_per_dim': 4, 'orientations': 6, 'downsample_factor': 8, 'num_words': 128}
        else:
            raise ValueError("Name datasets not recognized: " + name_datasets)
        X_train_described = describe_dataset(X_train, dict_call_feature[type_feature], params=feature_params)
        X_test_described = describe_dataset(X_test, dict_call_feature[type_feature], params=feature_params)
        if type_feature != list_feature[0] or type_feature == list_feature[3]:
            print('Saving described data')
            file = open(file_bin, 'wb')
            pickle.dump((X_train_described, X_test_described), file, pickle.HIGHEST_PROTOCOL)
            file.close()
    # get list c
    if type_feature == list_feature[0] or type_feature == list_feature[3]:  # raw_pixel and gray
        list_c = [0.0000001, 0.000001, 0.00001, 0.0001, 0.00025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
                  0.75, 1.0]
    elif type_feature == list_feature[1]:  # bow
        list_c = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.5,
                  3, 4, 5]
    elif type_feature == list_feature[2] or type_feature == list_feature[4]:  # hog and downs_hog
        if name_datasets == list_datasets[1]:  # cifar10
            list_c = [0.0001, 0.001, 0.0025, 0.005, 0.0075, 0.01,  0.025, 0.05, 0.075, 0.1, 0.5, 0.75, 1]
        else:
            list_c = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    else:
        raise ValueError("Feature is not implemented: " + type_feature)
    # return data and list c if it's for training
    if is_train:
        return X_train_described, X_test_described, list_c
    else:
        return X_train_described, X_test_described

# Execute svm
# ['raw_pixel', 'bow', 'hog', 'gray', 'downs_hog']
list_type_feature = [list_feature[4], list_feature[3]]
if __name__ == '__main__':
    for name_datasets in list_datasets:
        res_compute_svm_file = open('svm_result_' + name_datasets + '.txt', 'w')
        print('======= Loading data %s =======' % name_datasets)
        (X_train, y_train), (X_test, y_test) = loading_data(name_datasets)
        res_compute_svm_file.write('============== Datasets %s ==============\n' % name_datasets)
        # Training to find best params of svm
        for type_feature in list_type_feature:
            if (type_feature == list_feature[1] or type_feature == list_feature[3]) \
                    and name_datasets == list_datasets[0]:  # if mnist no raw_pixel in gray and bow
                continue
            elif type_feature == list_feature[2] and name_datasets == list_datasets[1]:  # if cifar10 no downs_hog
                continue
            # Decrease size because run time very long
            if type_feature in [list_feature[0], list_feature[4]] and name_datasets == list_datasets[1]:
                N_MAX = int(len(X_train) * 0.5)
                N_TRAIN = int(N_MAX * 0.7)
            else:
                N_MAX = len(X_train)
                N_TRAIN = int(len(X_train) * 0.7)
            X_train_train = X_train[:N_TRAIN]
            X_train_valid = X_train[N_TRAIN:N_MAX]
            y_train_train = y_train[:N_TRAIN]
            y_train_valid = y_train[N_TRAIN:N_MAX]

            print('=== Feature %s ===' % type_feature)
            res_compute_svm_file.write('===== Feature %s =====\n' % type_feature)
            print('=== Training ===')
            X_train_described, X_test_described, list_c = processing_data(X_train_train, X_train_valid,
                                                                          type_feature, name_datasets)
            described_data = ((X_train_described, y_train_train), (X_test_described, y_train_valid))

            res_grid = {list_loss[0]: [], list_loss[1]: []}
            best_hyper_params = {'C': 0.0, 'accuracy': 0.0, 'loss': None, 'diff_acc_abs': 100}
            for idx_c, c in enumerate(list_c):
                print('Training c=%s' % str(c))
                res = svm(dataset=described_data, c=c)
                for loss_type in list_loss:
                    res_grid[loss_type].append(res[loss_type][0])
                    print('Accuracy loss %s -|-  train = %.3f  -|- test = %.3f' % (loss_type,
                                                                                   res[loss_type][0]['train_accuracy'],
                                                                                   res[loss_type][0]['test_accuracy']))
                if best_hyper_params['accuracy'] < res[loss_type][0]['test_accuracy']\
                        and abs(res[loss_type][0]['train_accuracy'] - res[loss_type][0]['test_accuracy']) \
                        < best_hyper_params['diff_acc_abs']:
                    best_hyper_params['C'] = c
                    best_hyper_params['accuracy'] = res[loss_type][0]['test_accuracy']
                    best_hyper_params['loss'] = loss_type
                    best_hyper_params['diff_acc_abs'] = abs(res[loss_type][0]['train_accuracy']
                                                            - res[loss_type][0]['test_accuracy']) + 0.5
            # put result in csv
            list_val_name = ['C', 'train_accuracy', 'test_accuracy']
            res_file = open('svm_grid_search_accuracy_' + type_feature + '_' + name_datasets + '.txt', 'w')
            for loss_type in res_grid:
                res_file.write('Result %s;Train;Test;Diff;\n' % loss_type)
                print('\nResult %s:' % loss_type)
                for r in res_grid[loss_type]:
                    print(r)
                    line = ''
                    for v in list_val_name:
                        line += str(r[v]) + ';'
                    line += str(r[list_val_name[1]] - r[list_val_name[2]]) + ';'
                    res_file.write(line + '\n')
                res_file.write('\n')
            res_file.close()
            # Test svm with best params
            print('\n=== Test c=%s -|- accuracy_train_validation=%.3f -|- loss=%s ==='
                  % (str(best_hyper_params['C']), best_hyper_params['accuracy'], str(best_hyper_params['loss'])))
            X_train_described, X_test_described = processing_data(X_train, X_test, type_feature, name_datasets, False)
            described_data = ((X_train_described, y_train), (X_test_described, y_test))
            res = svm(dataset=described_data, c=best_hyper_params['C'])
            training_accuracy = res[best_hyper_params['loss']][0]['train_accuracy']
            test_accuracy = res[best_hyper_params['loss']][0]['test_accuracy']
            predicted_y = res[best_hyper_params['loss']][0]['predicted_y']

            # Compute confusion matrix
            cnf_matrix = compute_confusion_matrix(y_test, predicted_y, name_datasets, type_feature)
            print('c=%s;accuracy_cross_validation=%s;accuracy_training=%s;accuracy_test=%s;loss=%s\n'
                  % (str(best_hyper_params['C']), str(best_hyper_params['accuracy']), str(training_accuracy),
                     str(test_accuracy), best_hyper_params['loss']))
            res_compute_svm_file.write('c=%s;accuracy_cross_validation=%s;accuracy_training=%s;accuracy_test=%s;'
                                       'loss=%s\nmatrix_confusion\n'
                                       % (str(best_hyper_params['C']), str(best_hyper_params['accuracy']),
                                          str(training_accuracy), str(test_accuracy), best_hyper_params['loss']))
            res_compute_svm_file.write(str(cnf_matrix))
            res_compute_svm_file.write('\n')
        res_compute_svm_file.close()
