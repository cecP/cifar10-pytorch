
import matplotlib.pyplot as plt

import torch

import custom_models 
import load_cifar10 

# Loading the data
#------------------------------------------------------------------------------

use_gpu = True
PICKLED_FILES_PATH = "./Data/cifar-10-batches-py" 

X_train, y_train, X_test, y_test = load_cifar10.convert_pkl_to_numpy(PICKLED_FILES_PATH)

train_dataset = load_cifar10.CIFAR10Dataset(X_train, y_train, use_gpu)

# Training the model
#------------------------------------------------------------------------------

rnn_module = custom_models.RNNModule(3, 100, 2, 10, use_gpu)
model = custom_models.CustomModel(rnn_module, use_gpu)

# setting hyperparameters
batch_size = 400
learning_rate = 0.0001
num_epochs = 1
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

model.train(dataset=train_dataset,             
            batch_size=batch_size,
            loss=loss, 
            optimizer=optimizer, 
            num_epochs=num_epochs,
            val_batchsize=30)

torch.save(model.module.state_dict(), "rnn_model.params")

# Evaluation of the model
#------------------------------------------------------------------------------

# single image
current = X_test[2] 
plt.imshow(current)
model.predict(current[None,:], return_label=True) # should add an additional axis for the forward method to work

# multiple images
X_for_evaluation = X_test[0:100,:]
y_for_evaluation = y_test[0:100]
acc, cf = custom_models.predict_many_images(model, X_for_evaluation, y_for_evaluation)
print("Acc: {}, \n\nConfusion Matrix: \n {}".format(acc, cf))

