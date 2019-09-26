import gym
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F


env = gym.make('FrozenLake-v0')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 96)
        self.fc4 = nn.Linear(96, 96)
        self.fc5 = nn.Linear(96, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 4)

    def forward(self, x):
        x = Variable(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

model = Net()


def onehot2tensor(state):
    tmp = np.zeros(16)
    tmp[state] = 1
    vector = np.array(tmp, dtype='float32')
    tensor = torch.from_numpy(vector).float()
    return tensor

def applymodel(tensor):
    output_tensor = model(tensor)
    output_array = output_tensor.data.numpy()
    return output_tensor, output_array

total_reward = 0.0
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
steps = 10000

for i_episode in range(steps):
    observation = env.reset()
    episode_reward = 0.0
    total_loss = 0.0

    for t in range(100):
        current_state = observation
        optimizer.zero_grad()
        current_tensor = onehot2tensor(current_state)
        current_output_tensor, current_output_array = \
                applymodel(current_tensor)
        
        if np.random.rand() < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(current_output_array)

        observation, reward, done, info = env.step(action)
        observation_tensor = onehot2tensor(observation)
        observation_output_tensor, observation_output_array = \
                applymodel(observation_tensor)

        q = reward + 0.99 * np.max(observation_output_array)
        q_array = np.copy(current_output_array)
        q_array[action] = q
        q_variable = Variable(torch.Tensor(q_array))

        loss = criterion(current_output_tensor, q_variable)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()

        if done:
            episode_reward += reward

    total_reward += episode_reward

    if (i_episode + 1) % 1000 == 0:
        print(i_episode + 1, total_loss, total_reward)

print(total_reward / steps)

observation = env.reset()

for i_episode in range(100):
    env.render()
    current_state = observation
    current_tensor = onehot2tensor(current_state)
    current_output_tensor, current_output_array = \
            applymodel(current_tensor)
    action = np.argmax(current_output_array)
    observation, reward, done, info = env.step(action)
