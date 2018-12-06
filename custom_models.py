''' Model classes for pytorch along with some helper functions '''

import numpy as np

import torch
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
    
class CNNModule(nn.Module):
    ''' A module for convolutional neural network '''
    
    def __init__(self):
        super().__init__() 
        
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
    
    
class CNNModule2(nn.Module):
    ''' A module for convolutional neural network '''
    
    def __init__(self):
        super().__init__() 
        
        self.features = torch.nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=80, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(in_channels=80, out_channels=160, kernel_size=4, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(in_channels=160, out_channels=256, kernel_size=3, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2)
                    )     
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10)
        )
        
#        self.fc1 = nn.Linear(40 * 8 * 8, 10) # here it is easiest if you print the shape of the tensor before the linear layer in the forward method      

    def forward(self, x):     
        out = self.features(x) 

        out = out.view(out.size(0), -1) # the fully connected layer takes 1 dim input; 
                                        # the size of the batch is the first dimension of the new tensor and everything else flattened
         
        out = self.classifier(out)        
        
        return out
    
    

class RNNModule(nn.Module):
    ''' A module for recurrent neural network. It is only one directional. '''
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, use_gpu):
        super().__init__() 
        
        self.use_gpu = use_gpu
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.lstm_chan1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_chan2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_chan3 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Linear(3 * hidden_dim, output_dim)        
        
        
    def forward(self, x):   
        
        # Initialize hidden state - dimensions are (num_layers, batch_size, hidden_dim)
        h0_chan1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0_chan1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        h0_chan2 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0_chan2 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        h0_chan3 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0_chan3 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        if self.use_gpu:
            h0_chan1 = h0_chan1.cuda()
            c0_chan1 = c0_chan1.cuda()
            h0_chan2 = h0_chan2.cuda()
            c0_chan2 = c0_chan2.cuda()
            h0_chan3 = h0_chan3.cuda()
            c0_chan3 = c0_chan3.cuda()   
                
        # the pixel batches from each channel are taken separately
        pixels_batch_chan_1 = x[:,0,:,:]
        pixels_batch_chan_2 = x[:,1,:,:]
        pixels_batch_chan_3 = x[:,2,:,:]
        
        lstm_out_chan_1, _ = self.lstm_chan1(pixels_batch_chan_1, (h0_chan1, c0_chan1))
        lstm_out_chan_2, _ = self.lstm_chan1(pixels_batch_chan_2, (h0_chan2, c0_chan2))
        lstm_out_chan_3, _ = self.lstm_chan1(pixels_batch_chan_3, (h0_chan3, c0_chan3))
        
        lstm_out_chan_1 = lstm_out_chan_1[:, -1, :]
        lstm_out_chan_2 = lstm_out_chan_2[:, -1, :]
        lstm_out_chan_3 = lstm_out_chan_3[:, -1, :]
        
        lstm_out_concat = torch.cat([lstm_out_chan_1, lstm_out_chan_2, lstm_out_chan_3], 1)        
        
        out = self.fc(lstm_out_concat)                      
        
        return out
    
#------------------------------------------------------------------------------          
    
class CustomModel:     
    ''' Class that wraps nn.Modules into a model with predict and train functions '''    
    def __init__(self, Module, use_gpu):        
       self.module = Module  
       self.use_gpu = use_gpu
       if self.use_gpu:
           self.module.cuda()
    
    def predict(self, x, return_label=False):     
        ''' this method outputs the predictions in more convenient format than forward 
            x (numpy array, torch tensor) - pixels with shape (batch_size, 32, 32, 3)            
            return_label (boolean) - returns the name of the class not just the label code 
        '''      
        
        if not isinstance(x, torch.Tensor):
            x = np.moveaxis(x, 3, 1)                
            x = torch.tensor(x, dtype=torch.float32) 
        else:
            x = x.transpose(3, 1)
        
        if self.use_gpu:     
            x = x.cuda()  
                         
        outputs = self.module.forward(x)
        
        outputs = outputs.cpu() # cause cannot convert CUDA tensor to numpy array, the outputs should be sent to cpu
        outputs = outputs.data.numpy()
        predictions = np.argmax(outputs, 1)        
        
        if return_label:
            predictions = labels_dict[predictions[0]]
            
        return predictions
    
    @timer
    def train(self, loader, loss, optimizer, num_epochs): 
        ''' method that wraps the training of the model '''
        
        for epoch in range(num_epochs):
            print("epoch: {}".format(epoch))
            print("--------------------------------")
            for i, (pixels_batch, labels_batch) in enumerate(loader):
                optimizer.zero_grad()    
                if self.use_gpu:                    
                    pixels_batch = pixels_batch.cuda()
                    labels_batch = labels_batch.cuda()
                              
                outputs = self.module.forward(pixels_batch)    
                
                loss_tensor = loss(outputs, labels_batch)
                loss_tensor.backward()
                optimizer.step()
                if i % 100 == 0:
                    print("loss: {}".format(loss_tensor.data)) # for each epoch        
            print("loss: {}".format(loss_tensor.data)) # for each epoch
    

#------------------------------------------------------------------------------          
   
@timer       
def predict_many_images(model, X_test=None, y_test=None, dataset=None):
    ''' As memory problems may occur when trying to predict many images this function is a workaround for this 
    
        X_test (numeric numpy array): pixels with example shape (10000, 32, 32, 3)
        y_test (numeric numpy array): number corresponding to the labels
    '''
    
    prediction = []
    if dataset is None:
        num_images = X_test.shape[0]
        
        for i in range(num_images):   
            current = X_test[i][None,:]
            current_prediction = model.predict(current)
            prediction.append(current_prediction)
            if i % 100 == 0:
                print("{} of {}".format(i, num_images))
    else:
         num_images = len(dataset)
         y_test = []
         for i, (current, y) in enumerate(dataset):
             y_test.append(y)
             current = current.unsqueeze(0).transpose(3, 1)
             current_prediction = model.predict(current)
             prediction.append(current_prediction)
             if i % 100 == 0:
                 print("{} of {}".format(i, num_images))         
            
    prediction = np.vstack(prediction)
    prediction = np.squeeze(prediction)

    acc = np.mean(prediction == y_test)
    cf = sklearn.metrics.confusion_matrix(prediction, y_test)

    
    return acc, cf