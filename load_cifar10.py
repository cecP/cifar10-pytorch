'''
Code for importing and preparing the cifar10 dataset for modelling

The initial datafiles should be downloaded from here:
    https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 python version - should be downloaded and extracted in some folder
'''

import pickle
import numpy as np
import os

#------------------------------------------------------------------------------

def unpickle(file):
    # unpickles a file and returns a dictionary  
    # copied from the cifar10 site
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')        
    return dct

def reshape_dct_files(dct):
    # takes a dct file generated from unpickle and returns 2 numpy arrays
    # with pixels and with labels
    pixels_array = dct[b'data']
    pixels_array = pixels_array.reshape(pixels_array.shape[0], 3, 32, 32)
    pixels_array = np.moveaxis(pixels_array, 1, 3)
    
    labels_array = dct[b'labels']
    
    return pixels_array, labels_array
    
def convert_pkl_to_numpy(file_path):   
    # reshapes the pkl files to numpy arrays; creates two sets of arrays 
    # one for train and one for test
    # file_path is the path of the extracted cifar10 archive
    pixels_lists = []
    labels_lists = []
    
    for i in range(1, 6):
        dct = unpickle(os.path.join(file_path, 'data_batch_{}'.format(i)))
        pixels_array, labels_array = reshape_dct_files(dct)
        pixels_lists.append(pixels_array)
        labels_lists.append(labels_array)
    
    train_pixels_array = np.vstack(pixels_lists)
    train_labels_array = np.concatenate(labels_lists)
    
    dct = unpickle(os.path.join(file_path, 'test_batch'))
    test_pixels_array, test_labels_array = reshape_dct_files(dct)

    return train_pixels_array, train_labels_array, test_pixels_array, test_labels_array




#
#dct = unpickle("./Data/cifar-10-python/cifar-10-batches-py/data_batch_1")
#a[b"data"].shape
#
#
#np.array([]).reshape(32, 32, 3)
#
#plt.imshow(pixels_array[4]) 
#
#def load_CIFAR_batch(filename):
#    """ load single batch of cifar """
#    print(filename)
#    with open(filename, 'rb') as f:
#        datadict = cPickle.load(f)
#        X = datadict['data']
#        Y = datadict['labels']
#        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
#        Y = np.array(Y)
#        return X, Y
#
#def load_CIFAR10(ROOT):
#    """ load all of cifar """
#    xs = []
#    ys = []
#    for b in range(1, 6):
#        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
#        X, Y = load_CIFAR_batch(f)
#        xs.append(X)
#        ys.append(Y)    
#    Xtr = np.concatenate(xs)
#    Ytr = np.concatenate(ys)
#    del X, Y
#    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#    return Xtr, Ytr, Xte, Yte