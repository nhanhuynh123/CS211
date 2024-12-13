import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from make_env import make_env
from agent import Agent
class MAAC:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

        self.gamma = gamma
        self.tau = tau

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions



    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device  # Thiết bị của agent (CPU hoặc GPU)

        # Chuyển sang tensor và đưa vào device
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            # Lấy trạng thái mới của actor và tính giá trị từ mạng target
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)  # Mạng target của actor

            all_agents_new_actions.append(new_pi)

            # Lấy trạng thái cũ của actor và tính giá trị của hành động hiện tại
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)

            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        # Ghép các hành động mới của các agent
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # Tính giá trị critic từ mạng target critic
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0  # Thiết lập giá trị critic là 0 khi done
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + self.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            
            # Tính gradient và cập nhật mạng critic
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # Tính loss của actor
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)

            # Tính gradient và cập nhật mạng actor
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            # Cập nhật các tham số mạng target
            agent.update_network_parameters(self.tau)
