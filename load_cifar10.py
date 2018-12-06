''' Functions for importing and preparing the cifar10 dataset for modelling with pytorch

The initial datafiles are here: https://www.cs.toronto.edu/~kriz/cifar.html

As "CIFAR-10 python version" of the data is used, it should be downloaded and 
extracted in some folder. The tar.gz file contains pickled objects, which are 
processed by the code in this module.
'''

import pickle
import numpy as np
import os

import torch
import torch.utils.data 

#------------------------------------------------------------------------------

def timer(func):
    ''' decorator for timing functions '''
    import time
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()        
        print("{} runtime: {} seconds".format(func.__qualname__ , (stop - start))) # __qualname__ displays methods names as well as function names

        return result
    return wrapper

def unpickle(file):
    ''' Unpickles a file and returns a dictionary.  
    Copied from the cifar10 site.
    
    file is a pickled file from the cifar10 archive 
    '''
    
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')        
    return dct


def reshape_dct_files(dct):
    ''' Takes a dct file generated from unpickle and returns 2 numpy arrays with pixels and with labels ''' 
  
    pixels_array = dct[b'data']
    pixels_array = pixels_array.reshape(pixels_array.shape[0], 3, 32, 32)
    pixels_array = np.moveaxis(pixels_array, 1, 3)
    
    labels_array = dct[b'labels']
    
    return pixels_array, labels_array

@timer 
def convert_pkl_to_numpy(file_path): 
    ''' Reshapes the pkl files to numpy arrays. 
    Returns two sets of arrays one for train and one for test.
    
    file_path (string): the path of the extracted cifar10 archive
    '''

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
    test_labels_array = np.array(test_labels_array, "int32")
    
    return train_pixels_array, train_labels_array, test_pixels_array, test_labels_array

# pytorch specific classes for loading the data
#------------------------------------------------------------------------------

class CIFAR10Dataset(torch.utils.data.Dataset):    
    def __init__(self, pixels, labels, use_gpu, transform=None):   
        ''' A class used by the pytorch DataLoader for training of the models in pytorch.
        Should override __len__ and __getitem__ from Dataset
        
        Args:
            pixels (numeric numpy array): pixels with shape (batch_size, 32, 32, 3)
            labels (numeric numpy array): the labels codes 0 - 9
            transform (torchvision.transforms): optional transformations applied on the pixels 
        '''
        self.transform = transform
        self.pixels = pixels
        self.labels = labels
        self.use_gpu = use_gpu
                
        if pixels.shape[0] != labels.shape[0]:
            raise IOError('The length of the pixels and labels list does not match!')
                         
    def __len__(self):        
        return(len(self.labels))
        
    def __getitem__(self, idx):        
        pixels = self.pixels[idx]
        pixels = np.moveaxis(pixels, 2, 0)        

        label = self.labels[idx]
        
        if self.use_gpu:
            device_str = "cuda:0"
        else:
            device_str = "cpu"
        
        pixels = torch.tensor(pixels, dtype=torch.float, device=torch.device(device_str))
        label = torch.tensor(label, dtype=torch.long, device=torch.device(device_str))
        
        if self.transform:        
            pixels = self.transform(pixels) 
        
        return pixels, label

