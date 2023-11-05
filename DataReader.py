import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for batch in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{batch}'), 'rb') as file:
            batch_data = pickle.load(file, encoding='latin1')
            x_train.append(batch_data['data'])
            y_train += batch_data['labels']
    
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_train = np.array(x_train.reshape(-1,3072),dtype=np.float32)
    test_file = open(os.path.join(data_dir, 'test_batch'), 'rb')
    test_data = pickle.load(test_file, encoding='latin1')
    x_test.append(test_data['data'])
    y_test.append(batch_data['labels'])
    x_test=np.array(x_test,dtype=np.float32)
    x_test=x_test.reshape(-1,3072)
    y_test=np.array(y_test)
    print(x_train.shape)
    print(x_train.dtype)
    print(y_train.shape)
    print(y_train.dtype)
    print(x_test.shape)
    print(x_test.dtype)
    print(y_test.shape)
    print(y_test.dtype)
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid