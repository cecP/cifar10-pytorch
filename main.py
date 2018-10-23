
import matplotlib.pyplot as plt
from load_cifar10 import convert_pkl_to_numpy

#------------------------------------------------------------------------------

PICKLED_FILES_PATH = "./Data/cifar-10-python/cifar-10-batches-py" 

train_pixels_array, train_labels_array, test_pixels_array, test_labels_array = convert_pkl_to_numpy(PICKLED_FILES_PATH)

