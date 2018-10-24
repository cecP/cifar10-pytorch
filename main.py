
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from cnn_model import CNNModel
from load_cifar10 import convert_pkl_to_numpy, CIFAR10Dataset
#import ipdb

#------------------------------------------------------------------------------

PICKLED_FILES_PATH = "./Data/cifar-10-python/cifar-10-batches-py" 

X_train, y_train, X_test, y_test = convert_pkl_to_numpy(PICKLED_FILES_PATH)

# Training
#------------------------------------------------------------------------------

useGPU = False

train_dataset = CIFAR10Dataset(X_train, y_train)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=30,
                          shuffle=False)

model = CNNModel(useGPU)
if useGPU:
    model.cuda()
#model.load_state_dict(torch.load("model.params")) # if there are coefficients from pretrained model

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

start = timer()
model.train(loader=train_loader, loss=loss, optimizer=optimizer, num_epochs=1)
stop = timer()
#torch.save(model.state_dict(), "model.params")
print("Elapsed time: {}".format(stop - start))

# Evaluation
#------------------------------------------------------------------------------

current = X_test[1][None,:] # should add an additional axis for the forward method to work

prediction = model.predict(X_test[0:1000,:])
y = y_test[0:1000]
np.mean(prediction == y)

prediction = []
for i in range(X_test.shape[0]):   
    current = X_test[i][None,:]
    curr_prediction = model.predict(np.moveaxis(current, 3, 1))
    prediction.append(curr_prediction)
    if i % 100 == 0:
        print(i)
prediction = np.vstack(prediction)
prediction = np.squeeze(prediction)

np.mean(prediction == y_test_encoded)
cf = confusion_matrix(prediction, y_test_encoded)
