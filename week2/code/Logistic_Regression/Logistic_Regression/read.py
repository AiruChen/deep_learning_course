import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5',"r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    # your train set labels
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    # your test set labels
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # the list of classes
    classes = np.array(test_dataset["list_classes"][:])

    # print(train_set_y_orig.shape) # test, only read [209,]
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    # print(train_set_y_orig.shape) # test, reshape to [1,209]
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


