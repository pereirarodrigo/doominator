""" Convolutional neural network file.

This file contains the implementation of a convolutional neural network that
will be used as the actor and critic networks of the PPO algorithm.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    A convolutional neural network with 3 convolutional layers and 2 fully connected layers.
    """
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(256 * 8, 512)
        self.fc2 = nn.Linear(512, output_size)


    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float, device=self.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 256 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x