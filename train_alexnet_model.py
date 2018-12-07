
import torch
import torch.nn as nn

import torchvision.transforms as transforms

import custom_models 
import load_cifar10 

# Loading the data
#------------------------------------------------------------------------------

use_gpu = True
PICKLED_FILES_PATH = "./Data/cifar-10-batches-py" 

X_train, y_train, X_test, y_test = load_cifar10.convert_pkl_to_numpy(PICKLED_FILES_PATH)

# some transforms have to be applied as Alexnet cannot accept images less than 224x224
transform = transforms.Compose([
        transforms.ToPILImage(mode="RGB"), # input has to be converted to PIL image otherwise Resize won't work
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

train_dataset = load_cifar10.CIFAR10Dataset(X_train, y_train, use_gpu=False, transform=transform) # in order for thransforms to work output tensors should not be on gpu

# Training the model
#------------------------------------------------------------------------------

class AlexNet(nn.Module):
    ''' from pytorch github - https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py '''
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

alexnet = AlexNet(num_classes=10)

model = custom_models.CustomModel(alexnet, use_gpu)

# setting hyperparameters
batch_size = 200
learning_rate = 0.0001
num_epochs = 1
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

model.train(loader=train_loader, 
            loss=loss, 
            optimizer=optimizer, 
            num_epochs=num_epochs)

torch.save(model.module.state_dict(), "alexnet_model.params")

# Evaluation of the model
#------------------------------------------------------------------------------

X_for_evaluation = X_test[1000:2000,:]
y_for_evaluation = y_test[1000:2000]
test_dataset = load_cifar10.CIFAR10Dataset(X_for_evaluation, y_for_evaluation, transform=transform)
acc, cf = custom_models.predict_many_images(model, dataset=test_dataset)
print("Acc: {}, \n\nConfusion Matrix: \n {}".format(acc, cf))

