### Import Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class My2hl(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the model. The model is a simple feedforward neural network with 2 hidden layers.
        """
        super(My2hl, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = nnf.sigmoid(x)
        x = self.fc2(x)
        x = nnf.sigmoid(x)
        x = self.fc3(x)
        x = nnf.log_softmax(x, dim=1)
        return x


class My3hl(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the model. The model is a simple feedforward neural network with 3 hidden layers.
        """
        super(My3hl, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = nnf.sigmoid(x)
        x = self.fc2(x)
        x = nnf.sigmoid(x)
        x = self.fc3(x)
        x = nnf.sigmoid(x)
        x = self.fc4(x)
        x = nnf.log_softmax(x, dim=1)
        return x