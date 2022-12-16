# a CNN-RNN model based on BIN,
# (Wang H, Mao C, He H, et al. Bidirectional inference networks: A class of deep bayesian networks for health profiling[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 766-773.)


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import full_mse_loss
from src.args import parse_args
args = parse_args()



class CNNRNN (nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()
        width = args.width
        self.fc1 = nn.Linear(96, width)
        self.fc2 = nn.Linear(width, width)
        # self.fc3 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)

    def forward(self, x):
        target = x
        x = F.relu(self.fc1(target)) # U1
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x) # v2
        return x

