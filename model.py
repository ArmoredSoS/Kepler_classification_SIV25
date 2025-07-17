from utils import *
from function import init_weights

import torch.nn as nn   

class minimalCNN(nn.Module):
    """
    A barebone CNN, used to create a baseline for the tests
    """
    def __init__(self, input_ch=1, out_ch=16, kernel_size=3, stride=1, pool_kernel=2, output_dim=2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_ch, out_ch, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool1d(pool_kernel),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_ch, output_dim)
        )

    def forward(self, x):
        return self.conv(x)

class KeplerCNN(nn.Module):
    """
    CNN with improvements used for light curve analysis, improvements are:
        Dropout layers -> one after the convolutional layer and one before the linear
        Normalization layer -> after the dropout layer except before linear one
    """

    def __init__(self, activation, input_ch = 1, out_ch = 16, kernel_size = 3, 
                 stride = 1, pool_kernel = 2, output_dim = 2, 
                 dropout_rate = 0.3, init_weight = False):
        super().__init__()
        
        self.conv = nn.Sequential(
            
            #Convolutional layer
            nn.Conv1d(input_ch, out_ch, kernel_size, stride),
            nn.Dropout(p = dropout_rate), #Dropout layer after convolutional
            nn.BatchNorm1d(out_ch), #Normalization
            activation(), #Tunable activation function
            
            nn.MaxPool1d(pool_kernel),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p = dropout_rate), #Dropout layer before linear
            nn.Linear(out_ch, output_dim)
            
            # Additional normalization layer + linear + nonlinearity, dataset is too little, performs poorly:
            # nn.Linear(out_ch, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # activation(),
            # nn.Linear(hidden_dim, output_dim)
                
        )
        
        if init_weight: init_weights(self)
        
    def forward(self, x):
        output = self.conv(x)      
        return output

            
class OverComplicatedCNN_Test(nn.Module):
    """
    Name is self explanatory, gave worse result than the base CNN, just used as experiment
    """

    def __init__(self, input_ch = 1, out_ch = 16, kernel_size = 3, 
                 stride = 1, pool_kernel = 2, output_dim = 2, 
                 activation = nn.ReLU, dropout_rate = 0.3, proj = True):
        super().__init__()
        
        self.proj = proj
        if proj: self.projection = nn.Conv1d(input_ch, out_ch, kernel_size=1)
        
        self.conv = nn.Sequential(
            
            #1st convolutional layer
            nn.Conv1d(input_ch, out_ch, kernel_size, stride),
            # nn.Dropout(p = dropout_rate), #Dropout layer after convolutional
            nn.BatchNorm1d(out_ch), #Normalization
            activation(), #Tunable activation function
            
            #2nd convolutional layer
            nn.Conv1d(out_ch, out_ch, kernel_size, stride),
            nn.Dropout(p = dropout_rate),
            nn.BatchNorm1d(out_ch), 
            activation(), 
            
            nn.MaxPool1d(pool_kernel),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p = dropout_rate), #Dropout layer before linear
            nn.Linear(out_ch, output_dim)            
        )
        
        
    def forward(self, x):
        output = self.conv(x)

        #Residual connection with projection to ensure match in size
        if self.proj: 
            if(x.shape == output.shape): return x + output 
            return self.projection(x) + output
        
        return output