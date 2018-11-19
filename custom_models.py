''' Model classes for pytorch along with some helper functions '''

import numpy as np

import torch as torch
import torch.nn as nn
import sklearn.metrics

from load_cifar10 import timer
#import ipdb



labels_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
        }



class CNNModel(nn.Module):
    ''' A model class for convolutional neural network '''
    
    def __init__(self, useGPU):
        super(CNNModel, self).__init__() 
        self.useGPU = useGPU        
        self.cnn_model = torch.nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=80, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(in_channels=80, out_channels=40, kernel_size=4, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )        
        self.fc1 = nn.Linear(40 * 8 * 8, 10) # here it is easiest if you print the shape of the tensor before the linear layer in the forward method      

    def forward(self, x):   
        out = self.cnn_model(x) 
        out = out.view(out.size(0), -1) # the fully connected layer takes 1 dim input; 
                                        # the size of the batch is the first dimension of the new tensor and everything else flattened
           
        out = self.fc1(out)        
        
        return out
    
         
    def predict(self, x, return_label=False):     
        ''' this method outputs the predictions in more convenient format than forward 
            x (numpy array) - pixels with shape (batch_size, 32, 32, 3)
            return_label (boolean) - returns the name of the class not just the label code 
        '''      
        x = np.moveaxis(x, 3, 1)                
        x = torch.tensor(x, dtype=torch.float32) 
        
        if self.useGPU:     
            x = x.cuda()                              
        outputs = self.forward(x)
        
        outputs = outputs.cpu() # cause cannot convert CUDA tensor to numpy array, the outputs should be sent to cpu
        outputs = outputs.data.numpy()
        predictions = np.argmax(outputs, 1)        
        
        if return_label:
            predictions = labels_dict[predictions[0]]
            
        return predictions
    
    @timer
    def train(self, loader, loss, optimizer, num_epochs):        
        for epoch in range(num_epochs):
            for i, (pixels, labels) in enumerate(loader):
                optimizer.zero_grad()    
                if self.useGPU:                    
                    pixels = pixels.cuda()
                    labels = labels.cuda()

                outputs = self.forward(pixels)    
                
                loss_tensor = loss(outputs, labels)
                loss_tensor.backward()
                optimizer.step()
                if i % 100 == 0:
                    print("loss: {}".format(loss_tensor.data)) # for each epoch        
            print("loss: {}".format(loss_tensor.data)) # for each epoch
            
#------------------------------------------------------------------------------          
   
@timer       
def predict_many_images(model, X_test, y_test):
    ''' As memory problems may occur when trying to predict many images this function is a workaround for this 
    
        X_test (numeric numpy array): pixels with example shape (10000, 32, 32, 3)
        y_test (numeric numpy array): number corresponding to the labels
    '''
    prediction = []
    num_images = X_test.shape[0]
    
    for i in range(num_images):   
        current = X_test[i][None,:]
        current_prediction = model.predict(current)
        prediction.append(current_prediction)
        if i % 100 == 0:
            print("{} of {}".format(i, num_images))
    prediction = np.vstack(prediction)
    prediction = np.squeeze(prediction)

    acc = np.mean(prediction == y_test)
    cf = sklearn.metrics.confusion_matrix(prediction, y_test)

    return acc, cf