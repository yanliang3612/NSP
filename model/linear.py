import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Linear (nn.Module):
    def __init__(self,input_size,output_size):
        super(Linear, self).__init__()

        self.liner = nn.Linear(in_features= input_size, out_features=output_size)

    def forward(self, x):
        out = self.liner(x)
        out = out.unsqueeze(dim=0)
        return out
