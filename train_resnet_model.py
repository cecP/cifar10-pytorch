
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchsummary 

import custom_models 
import load_cifar10
import resnet_module

# Loading the data
#------------------------------------------------------------------------------

use_gpu = False
PICKLED_FILES_PATH = "./Data/cifar-10-batches-py" 

X_train, y_train, X_test, y_test = load_cifar10.convert_pkl_to_numpy(PICKLED_FILES_PATH)

# some transforms have to be applied to accept images less than 224x224
transform = transforms.Compose([
        transforms.ToPILImage(mode="RGB"), # input has to be converted to PIL image otherwise Resize won't work
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

train_dataset = load_cifar10.CIFAR10Dataset(X_train, y_train, use_gpu=False, transform=transform) # in order for thransforms to work output tensors should not be on gpu

#------------------------------------------------------------------------------

resnet = resnet_module.ResNet(resnet_module.BasicBlock, [2, 2, 2, 2])

model = custom_models.CustomModel(resnet, use_gpu)

# setting hyperparameters
batch_size = 200
learning_rate = 0.0001
num_epochs = 1
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)

model.train(dataset=train_dataset,             
            batch_size=batch_size,
            loss=loss, 
            optimizer=optimizer, 
            num_epochs=num_epochs,
            val_batchsize=30)

torch.save(model.module.state_dict(), "resnet_model.params")

# Evaluation of the model
#------------------------------------------------------------------------------

X_for_evaluation = X_test[1000:2000,:]
y_for_evaluation = y_test[1000:2000]
test_dataset = load_cifar10.CIFAR10Dataset(X_for_evaluation, y_for_evaluation, use_gpu=False, transform=transform)
acc, cf = custom_models.predict_many_images(model, dataset=test_dataset)
print("Acc: {}, \n\nConfusion Matrix: \n {}".format(acc, cf))







