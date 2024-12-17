import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

from network import DQNNetwork

# device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

# class DQNNetwork(nn.Module):
#     def __init__(self, state_dims, action_dims):
#         super(DQNNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dims, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.q_values = nn.Linear(64, action_dims)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x_ = F.relu(self.fc2(x))
#         q_values = self.q_values(x_)
#         return q_values

class DQNAgent:
    def __init__(self, state_dims, action_dims, agent_name, gamma=0.99, lr=0.001, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.agent_name = agent_name
        self.q_network = DQNNetwork(state_dims, action_dims, lr)
        self.target_network = DQNNetwork(state_dims, action_dims, lr)
        self.action_dims = action_dims

        self.update_target_network()

        self.device = self.q_network.device

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, epsilon):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device)
        with T.no_grad():
            if np.random.rand() < epsilon:
                return np.random.choice(self.action_dims)
            else:
                q_values = self.q_network(state)
                return T.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        # Chuyển đổi kiểu của s, s' sang tensor
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float).unsqueeze(0).to(self.device)

        # No grad computation
        with T.no_grad():
            next_q_values, _ = self.target_network(next_state).max(dim=2)
            next_q_values = next_q_values.squeeze()
        # Compute Q_value
        q_values = self.q_network(state).squeeze(0).gather(1, action).squeeze(0).flatten()
        # Compute target
        target = reward + ~done * self.gamma * next_q_values
        # Compute loss
        loss = (q_values - target).pow(2).mean()
        # Update params
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

