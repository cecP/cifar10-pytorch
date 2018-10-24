'''
Code for importing and preparing the cifar10 dataset for modelling

The initial datafiles should be downloaded from here:
    https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 python version - should be downloaded and extracted in some folder
'''

import pickle
import numpy as np
import os

import torch
from torch.utils.data import Dataset

#------------------------------------------------------------------------------

def unpickle(file):
    ''' 
    Unpickles a file and returns a dictionary.  
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

 
def convert_pkl_to_numpy(file_path): 
    ''' 
    Reshapes the pkl files to numpy arrays. 
    Returns two sets of arrays one for train and one for test.
    
    file_path: the path of the extracted cifar10 archive
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

class CIFAR10Dataset(Dataset):    
    def __init__(self, pixels, labels, transform=None):   
        '''
        A class used by the pytorch DataLoader for training of the models in pytorch
        
        Args:
            pixels (int numpy array): pixels with shape (batch_size, 32, 32, 3)
            labels (int numpy array): the labels codes 0 - 9
            transform: optional transformations applied on the pixels 
        '''
        self.transform = transform
        self.pixels = pixels
        self.labels = labels
                
        if pixels.shape[0] != labels.shape[0]:
            raise IOError('The length of the pixels and labels list does not match!')
                         
    def __len__(self):        
        return(len(self.labels))
        
    def __getitem__(self, idx):        
        pixels = self.pixels[idx]
        pixels = np.moveaxis(pixels, 2, 0)        

        label = self.labels[idx]
        
        pixels = torch.tensor(pixels, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:        
            pixels = self.transform(pixels) 
        
        return pixels, label

