
# Pytorch models implemented on CIFAR10

Deep learning models for CIFAR10 implemented in pytorch. CNN, one directional RNN and Alexnet implemented up till now. 
Useful for testing the performance of different model architectures. Can run both on CPU only and GPU, but CPU is too slow for
the more complicated models.


### Prerequisites

Python environment with pytorch, torchvision and scikit-learn is required. 

### Getting the Data
Download the python version of the CIFAR10 dataset from the official website: https://www.cs.toronto.edu/~kriz/cifar.html. 
It contains an archive with pickle files. In **load_data.py** one can find functions to load the data from the pickle files into a pytorch Dataset. 

<br/>
<br/>

## Models

### CNN models
Use the code **train_cnn_model.py**. 
Some architectures are present in **custom_models.py**. To implement a new architecture one must create a class inheriting nn.Module 
and implementing *\_\_init__* and *forward* methods. Accuracy is evaluated with confusion matrix and percentage of correct hits. 

### RNN models
The same as CNN, but the code is **train_rnn_model.py**. Bidirectional RNN would be the next model applied when I have time.

### Prebuilt pytorch models
Some prebuilt model architectures can be found here: https://github.com/pytorch/vision/tree/master/torchvision/models.

The code in **train_alexnet_model.py** implements the AlexNet architecture from the link above. 
