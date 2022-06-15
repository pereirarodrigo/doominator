""" Convolutional neural network file.

This file contains the implementation of a convolutional neural network that
will be used as the actor and critic networks of the PPO algorithm.

@author: Rodrigo Pereira
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    A convolutional neural network with 3 convolutional layers and a fully connected layer.
    """
    def __init__(self, c, h, w, output_size):
        super(CNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten()
        )

        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

        self.net = nn.Sequential(
            self.net,
            nn.Linear(self.output_dim, output_size),
            nn.ReLU(inplace=True)
        )

        self.output_dim = output_size


    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        logits = self.net(obs)
        
        return logits, state