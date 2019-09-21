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

wine_data   = wine.data[:130]
wine_target = wine.target[:130]

#  print(wine_data)
#  print(wine_target)

train_X, test_X, train_Y, test_Y = train_test_split(wine_data, wine_target, test_size=0.2)

#  print(len(train_X))
#  print(len(test_X))

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()
test_X  = torch.from_numpy(test_X).float()
test_Y  = torch.from_numpy(test_Y).long()

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=16, shuffle=True)

#  print(train[:5])
#  print(train_loader)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 96)
        self.fc4 = nn.Linear(96, 96)
        self.fc5 = nn.Linear(96, 96)
        self.fc6 = nn.Linear(96, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x)

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()

    if (epoch+1) % 50 == 0:
        print(epoch+1, '\t',total_loss)


test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(accuracy)
