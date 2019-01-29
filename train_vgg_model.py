
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import custom_models 
import load_cifar10
import torchsummary  

# Loading the data
#------------------------------------------------------------------------------

use_gpu = True
PICKLED_FILES_PATH = "./Data/cifar-10-batches-py" 

X_train, y_train, X_test, y_test = load_cifar10.convert_pkl_to_numpy(PICKLED_FILES_PATH)

# some transforms have to be applied to accept images less than 224x224
transform = transforms.Compose([
        transforms.ToPILImage(mode="RGB"), # input has to be converted to PIL image otherwise Resize won't work
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

train_dataset = load_cifar10.CIFAR10Dataset(X_train, y_train, use_gpu=False, transform=transform) # in order for thransforms to work output tensors should not be on gpu

# Testing outputs from layers
#------------------------------------------------------------------------------

class VGG_net(nn.Module):
    ''' from pytorch github - https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py '''
    
    def __init__(self, num_classes=10, init_weights=True):
        super(VGG_net, self).__init__()
        
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
          )
                        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),     
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

vggnet = VGG_net(num_classes=10)
torchsummary.summary(vggnet, input_size=(3, 224, 224), batch_size=50, device="cpu")

model = custom_models.CustomModel(vggnet, use_gpu)

# setting hyperparameters
batch_size = 50
learning_rate = 0.0001
num_epochs = 3
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

#torch.save(model.module.state_dict(), "vggnet_model.params")

# Evaluation of the model
#------------------------------------------------------------------------------

X_for_evaluation = X_test[1000:2000,:]
y_for_evaluation = y_test[1000:2000]
test_dataset = load_cifar10.CIFAR10Dataset(X_for_evaluation, y_for_evaluation, use_gpu=False, transform=transform)
acc, cf = custom_models.predict_many_images(model, dataset=test_dataset)
print("Acc: {}, \n\nConfusion Matrix: \n {}".format(acc, cf))

