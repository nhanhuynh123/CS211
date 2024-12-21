import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# T.autograd.set_detect_anomaly(True)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, action_dims, name, chkpt_dir):

        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.state_net = nn.Sequential(
            # lớp 1
            nn.Conv2d(3, 32, 8, stride=4), # 84x84 -> 20x20
            nn.ReLU(),
            # lớp 2
            nn.Conv2d(32, 64, 4, stride=2), # 20x20 ->9x9
            nn.ReLU(),
            # lớp 3
            nn.Conv2d(64, 64, 3, stride=1), # 9x9 -> 7x7
            nn.ReLU(), # Chuyển từ 4D tensor -> 2D tensor
            # Fully conected layers
            nn.Flatten(), # đầu ra 64x7x7=3136
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        pi = self.state_net(state/255.0)
        pi = F.tanh(pi)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.state_net = nn.Sequential(
            # lớp 1
            nn.Conv2d(3, 32, 8, stride=4), # 84x84 -> 20x20
            nn.ReLU(),
            # lớp 2
            nn.Conv2d(32, 64, 4, stride=2), # 20x20 ->9x9
            nn.ReLU(),
            # lớp 3
            nn.Conv2d(64, 64, 3, stride=1), # 9x9 -> 7x7
            nn.ReLU(), # Chuyển từ 4D tensor -> 2D tensor
            # Fully conected layers
            nn.Flatten(), # đầu ra 64x7x7=3136
        )
        self.action_net = nn.Sequential(
            nn.Linear(n_actions, 128),
            nn.ReLU()
        )

        self.q_net = nn.Sequential(
            nn.Linear(3136 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state = state.permute(0, 3, 1, 2)
        state_ = self.state_net(state/255.0)
        action_ = self.action_net(action)

        combined = T.cat([state_, action_], dim=-1)

        q = self.q_net(combined)
        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))