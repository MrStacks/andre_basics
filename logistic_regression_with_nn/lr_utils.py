import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r") # to see files in train_dataset call train_dataset.keys()
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # saves train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # saves train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # saves test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # saves test set labels

    classes = np.array(test_dataset["list_classes"][:]) # saves the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes