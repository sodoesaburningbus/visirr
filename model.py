### This class file contains the code for the irrigation detection Neural Network
### Christopher Phillips

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class irr_net(nn.Module):

    def __init__(self):

        # Initialize the network class
        super(irr_net, self).__init__()

        # Define the desired layers
        n_outs = 3
        self.conv1 = nn.Conv1d(1,n_outs,3,padding=0)

        # Define layer that merges the inputs
        self.dense1 = nn.Linear(68,34)
        self.dense2 = nn.Linear(34,16)
        self.dense3 = nn.Linear(16,8)
        self.dense4 = nn.Linear(8,4)
        self.dense5 = nn.Linear(4,1)

        return
    
    def forward(self, X):

        # Split off the temperature data
        t = torch.unsqueeze(X[:,2:], dim=1)
        s = X[:,:2]

        # Perform convolution
        t = F.relu(self.conv1(t))
        t = torch.flatten(t, start_dim=1)

        # Now re-combine data
        X = torch.cat((s,t), axis=1)

        # Feed into dense layers
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = F.relu(self.dense3(X))
        X = F.relu(self.dense4(X))
        X = self.dense5(X)

        return X