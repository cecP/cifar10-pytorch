
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import torch

import custom_models 
import load_cifar10 
#import ipdb

# Loading the data
#------------------------------------------------------------------------------

PICKLED_FILES_PATH = "./Data/cifar-10-python/cifar-10-batches-py" 

X_train, y_train, X_test, y_test = load_cifar10.convert_pkl_to_numpy(PICKLED_FILES_PATH)
train_dataset = load_cifar10.CIFAR10Dataset(X_train, y_train)

# Training the model
#------------------------------------------------------------------------------

useGPU = False
model = custom_models.CNNModel(useGPU)
if useGPU:
    model.cuda()
#model.load_state_dict(torch.load("model.params")) # if coefficients from pretrained model would be used

# setting hyperparameters
batch_size = 30
learning_rate = 0.0001
num_epochs = 2
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
start = timer()
model.train(loader=train_loader, 
            loss=loss, 
            optimizer=optimizer, 
            num_epochs=num_epochs)
stop = timer()
print("Elapsed time: {}".format(stop - start))
torch.save(model.state_dict(), "model.params")


# Evaluation of the model
#------------------------------------------------------------------------------

# single image
current = X_test[1] 
plt.imshow(current)
model.predict(current[None,:], return_label=True) # should add an additional axis for the forward method to work

# multiple images
X_for_evaluation = X_test[0:1000,:]
y_for_evaluation = y_test[0:1000]
acc, cf = custom_models.predict_many_images(model, X_for_evaluation, y_for_evaluation)
print("Acc: {}, \n\nConfusion Matrix: \n {}".format(acc, cf))

