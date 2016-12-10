import numpy as np
import time
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix

import utils

np.random.seed(1337)  # for reproducibility
NB_CLASSES = 10


def preprocess_mnist():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = utils.flatten_dataset(X_train)
    X_test = utils.flatten_dataset(X_test)
    input_dim = X_train.shape[1]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return (X_train, Y_train), (X_test, Y_test), input_dim


def preprocess_cifar10():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = utils.flatten_dataset(X_train)
    X_test = utils.flatten_dataset(X_test)

    input_dim = X_train.shape[1]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return (X_train, Y_train), (X_test, Y_test), input_dim


def build_network(layer_size, input_dim):
    # Build convolution network
    model = Sequential()
    model.add(Dense(layer_size[0], input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(layer_size[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(layer_size[2]))
    model.add(Activation('relu'))
    model.add(Dense(layer_size[3]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # model.add(Flatten())
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
(X_train, Y_train), (X_test, Y_test), input_dim = preprocess_cifar10()

# Grid search
list_layer_size = [[64, 64, 64, 64], [128, 128, 128, 128], [64, 64, 128, 128], [64, 128, 64, 128]]
list_batch_size = [8, 16, 32, 64, 128]

grid_search_results = np.zeros((len(list_layer_size), len(list_batch_size)))

with open("cumulative_results.txt", "a") as file:
    file.write("\n\nNew grid search ==== " + time.ctime() + "\n")
    file.write("Nb. epochs: {}\n".format(nb_epoch))

for i, layer_size in enumerate(list_layer_size):
    # Build network
    model = build_network(layer_size, input_dim=input_dim, )
    model.summary()

    model.save_weights("start.hdf5")
    best_params = {'layer': None, 'batch': None, 'acc_train': 0}
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
        if accuracy > best_params['acc_train']:
            best_params = {'layer': layer_size, 'batch': batch_size, 'acc_train': accuracy}

        with open("cumulative_results.txt", "a") as file:
            file.write("{:<20} {:<5} {:<.4f} {:<.4f}\n".format(str(layer_size), batch_size, loss, accuracy))

        np.savetxt("gridsearch.csv", grid_search_results, delimiter="\t")


model = build_network(best_params['layer'], input_dim=input_dim, )
model.summary()
# Train
model.fit(X_train, Y_train, batch_size=best_params['batch'], nb_epoch=nb_epoch,
          verbose=1, validation_split=1 / 7)
# Test
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names[0], loss)
print(model.metrics_names[1], accuracy)

with open("cumulative_results.txt", "a") as file:
    file.write("\n\nResult best params\n")
    file.write("{:<20} {:<5} {:<.4f} {:<.4f}\n".format(str(best_params['layer']), best_params['batch'], loss, accuracy))

print('Compute confusion matrix')
predicted_y = model.predict(X_test)
cnf = np.zeros([NB_CLASSES, NB_CLASSES])
for i, pred in enumerate(predicted_y):
    predicted_class = np.argmax(pred)
    real_class = np.argmax(Y_test[i])
    cnf[real_class][predicted_class] += 1
np.savetxt("cnf_matrix.csv", cnf)
