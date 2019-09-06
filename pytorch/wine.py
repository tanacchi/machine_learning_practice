import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import pandas as pd

wine = load_wine()
#  print(wine)
#  print(pd.DataFrame(wine.data, columns=wine.feature_names))
#  print(wine.target)
