

%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import ipdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from MNISTMineDataset import MNISTMineDataset, ReshapeToPic

from RNN import RNNmodel

# MNIST import
exec(open("MNIST_data_import.py").read())

#------------------------------------------------------------------------------

train_dataset = MNISTMineDataset(mnist_train, transform=ReshapeToPic((28, 28), withChannel=False))

#type(train_dataset.__getitem__(1)[0])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=20,
                          shuffle=False)

# should transform

model = RNNmodel(28, 100, 2, 10)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train(loader=train_loader, loss=loss, optimizer=optimizer, num_epochs=10)

model.evaluate(np.reshape(mnist_test_data, [10000, 28, 28]), mnist_test_labels)

# model evaluation
#------------------------------------------------------------------------------        

predictions = model.predict(np.reshape(mnist_test_data, [10000, 28, 28]))
indexes_of_mismatches = np.where(predictions != mnist_test_labels)[0] # no idea why this is a tuple

index = indexes_of_mismatches[5]
plot(mnist_test_data[index])
model.predict(np.reshape(mnist_test_data[index], [1, 28, 28]))

np.mean(predictions == mnist_test_labels)

confusion_matrix(predictions, mnist_test_labels)



# on custom images
#------------------------------------------------------------------------------

folder = "D:/62 Image Recognition 02/MNIST/data/MNISTcsvs/"
name = "A0 - Copy.jpg"

im1 = Image.open(folder + name)
im1 = im1.resize((28, 28))
im1 = im1.convert('L') # converts to grayscale
im1 = im1.getdata()
im1 = 255 - np.array(im1).astype(np.float32)

plot(im1)
a = np.reshape(im1, (1, 28, 28))
model.predict(a)



# plotting the weights
#------------------------------------------------------------------------------

list(model.parameters())[0][0]
weights = list(model.parameters())[0].data.numpy()
plot(weights[0])
