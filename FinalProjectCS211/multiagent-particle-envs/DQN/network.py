import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, state_dims, action_dims, lr):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_values = nn.Linear(64, action_dims)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x_ = F.relu(self.fc2(x))
        q_values = self.q_values(x_)

        return q_values
