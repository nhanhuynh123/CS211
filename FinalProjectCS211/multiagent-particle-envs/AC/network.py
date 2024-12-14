import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, lr):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x_ = F.relu(self.fc2(x))
        aciton_probs = T.softmax(self.pi(x_), dim=-1)
        
        return aciton_probs

class Critic(nn.Module):
    def __init__(self, state_dims, lr):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x_ = F.relu(self.fc2(x))
        value = self.value(x_)

        return value