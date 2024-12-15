import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class DQNNetwork(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_values = nn.Linear(64, action_dims)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x_ = F.relu(self.fc2(x))
        q_values = self.q_values(x_)
        return q_values

class DQNAgent:
    def __init__(self, state_dims, action_dims, gamma=0.99, lr=0.001, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = DQNNetwork(state_dims, action_dims).to(device)
        self.target_network = DQNNetwork(state_dims, action_dims).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.action_dims = action_dims
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(device)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dims)
        else:
            q_values = self.q_network(state)
            return T.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(device)
        next_state = T.tensor(next_state, dtype=T.float).unsqueeze(0).to(device)
        reward = T.tensor(reward, dtype=T.float).unsqueeze(0).to(device)
        done = T.tensor(done, dtype=T.float).unsqueeze(0).to(device)

        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)
        target = reward + (1 - done) * self.gamma * T.max(next_q_values, dim=1)[0]
        q_value = q_values[0, action]

        loss = (q_value - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class MultiAgentDQN:
    def __init__(self, obs_spaces, action_spaces, gamma=0.99, lr=0.001, epsilon=0.1):
        self.agents = []
        for obs, act in zip(obs_spaces, action_spaces):
            self.agents.append(DQNAgent(obs.shape[0], act.n, gamma, lr, epsilon))

    def choose_actions(self, observations):
        actions = []
        for idx, agent in enumerate(self.agents):
            actions.append(agent.choose_action(observations[idx]))
        return actions

    def learn(self, experiences):
        for idx, (state, action, reward, next_state, done) in enumerate(experiences):
            self.agents[idx].learn(state, action, reward, next_state, done)
        for agent in self.agents:
            agent.update_target_network()
