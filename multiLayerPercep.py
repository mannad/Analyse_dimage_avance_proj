import numpy as np
import time
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility
NB_CLASSES = 10


def preprocess_mnist():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return (X_train, Y_train), (X_test, Y_test), input_shape


def preprocess_cifar10():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    input_shape = (X_train.shape[1], X_train.shape[2], 3)  # RGB

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return (X_train, Y_train), (X_test, Y_test), input_shape


def build_network(layer_size, input_shape):
    # Build convolution network
    model = Sequential()

    model.add(Dense(layer_size[0]))
    model.add(Activation('relu'))
    model.add(Dense(layer_size[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(layer_size[2]))
    model.add(Activation('relu'))
    model.add(Dense(layer_size[3]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


nb_epoch = 10

# Preprocess dataset
(X_train, Y_train), (X_test, Y_test), input_shape = preprocess_cifar10()

# Grid search
list_layer_size = [[64, 64, 64, 64], [128, 128, 128, 128], [64, 64, 128, 128], [64, 128, 64, 128]]
list_batch_size = [8, 16, 32, 64, 128]

grid_search_results = np.zeros((len(list_layer_size), len(list_batch_size)))

with open("cumulative_results.txt", "a") as file:
    file.write("\n\nNew grid search ==== " + time.ctime() + "\n")
    file.write("Nb. epochs: {}\n".format(nb_epoch))

for i, layer_size in enumerate(list_layer_size):
    # Build network
    model = build_network(layer_size, input_shape=input_shape,)
    model.summary()

    model.save_weights("start.hdf5")

    for j, batch_size in enumerate(list_batch_size):
        print("Training with:", layer_size, batch_size)

        model.load_weights("start.hdf5")

        # Train
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_split=1 / 7)

        # Test
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

        print(model.metrics_names[0], loss)
        print(model.metrics_names[1], accuracy)

        grid_search_results[i, j] = accuracy

        with open("cumulative_results.txt", "a") as file:
            file.write("{:<20} {:<5} {:<.4f} {:<.4f}\n".format(str(layer_size), batch_size, loss, accuracy))

        np.savetxt("gridsearch.csv", grid_search_results, delimiter="\t")
