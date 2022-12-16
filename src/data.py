import numpy as np
import sklearn
import pandas as pd
import scipy
import torch
from src.args import parse_args
from src.utils import set_random_seeds

args = parse_args()





# loading data
def load_data():
    data_native = pd.read_csv("/root/NSP/data/data_x.csv")
    order = []
    for i in range(5)[1:]:
        for j in range(24):
          order.append(str((i + j * 4)))
    data_native = data_native[order]
    data_x = np.array(data_native).astype('float32')
    data_y = np.array(pd.read_csv("/root/NSP/data/data_y.csv", header=None)).astype('float32')
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y).reshape(-1)
    return data_x,data_y




# splitting data,and transform data to torch tensor

def data_split(data_x,data_y,train_num):
    long_index= data_y.shape[0]
    cls_idex = torch.randperm(long_index)
    train_idex = cls_idex[:train_num]
    test_idex  = cls_idex[train_num:]
    train_x, test_x  = data_x[train_idex],data_x[test_idex]
    train_y,test_y = data_y[train_idex],data_y[test_idex]
    return train_x,train_y,test_x,test_y











