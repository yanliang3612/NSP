import os
import os.path as osp
import random
import numpy as np
import torch
import statistics



def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    np.random.seed(random_seed)
    random.seed(random_seed)




def summary(loss,repetitions):
    test_loss_mean = statistics.mean(loss)
    test_loss_std = (statistics.stdev(loss) / (repetitions ** (1 / 2)))
    log = 'test_loss_mean:{:.2f}, test_loss_std:{:.2f}'.format(test_loss_mean, test_loss_std)
    return log

