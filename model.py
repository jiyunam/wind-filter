'''
    Write a model for gesture classification.
'''
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()

        self.input_size = input_size
        self.output_size = 1    # i.e. good vs. bad
        self.fc1_size = 64

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)    # first fully connected layer
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(self.fc1_size, self.output_size)
        self.act2 = nn.Sigmoid()


    def forward(self, features):
        x = self.fc1(features)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)

        return x