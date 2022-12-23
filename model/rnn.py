# a CNN-RNN model based on BIN,
# (Wang H, Mao C, He H, et al. Bidirectional inference networks: A class of deep bayesian networks for health profiling[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 766-773.)

import torch.utils.data
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from src.args import parse_args
args = parse_args()



class RNN (nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.liner = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, self.hidden_size)
        out = self.liner(out)
        out = out.unsqueeze(dim=0)
        return out


