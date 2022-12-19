import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  
        self.linear1 = nn.Linear(hidden_size, output_size)  

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s,  h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s , h)
        x = self.linear1(x)
        x = x.view(s , -1)
        return x








