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

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is reshaped to a 2D tensor and passed through the layers.
        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = x.view(-1, 28 * 28)
        _x = self.fc1(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc2(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc3(_x)
        _x = nnf.log_softmax(_x, dim=1)
        return _x


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

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is reshaped to a 2D tensor and passed through the layers.
        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = x.view(-1, 28 * 28)
        _x = self.fc1(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc2(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc3(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc4(_x)
        _x = nnf.log_softmax(_x, dim=1)
        return _x