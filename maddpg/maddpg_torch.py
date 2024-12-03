import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from make_env import make_env
from agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
            scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64,
            gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.noise_scale = 0.1
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                gamma=gamma, tau=tau, chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    def choose_noise_action(self, obs, agent_index):
        
        actions = self.agents[agent_index].choose_action(obs)
        noise = np.random.uniform(-self.noise_scale, self.noise_scale, size=len(actions))

        noise_actions = actions + noise
        noise_actions = np.clip(noise_actions, 0.0, 1.0)
        noise_actions = noise_actions.astype(np.float32)
        return noise_actions
    
    def choose_unnoise_action(self, obs, agent_index):
        actions = self.agents[agent_index].choose_action(obs)
        return actions
    
    def learn(self, states, noise_actions, free_noise_actions, rewards, states_, actions_, dones, id):

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        noise_actions = T.tensor(noise_actions, dtype=T.float).to(device)
        free_noise_actions = T.tensor(free_noise_actions, dtype=T.float).to(device)
        actions_ = T.tensor(actions_, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        
        old_actions = T.cat([acts for acts in noise_actions], dim=1)
        mu_actions = T.cat([acts for acts in free_noise_actions], dim=1)
        next_actions = T.cat([acts for acts in actions_], dim=1)

        # Update critic
        critic_value_ = (self.agents[id].target_critic.forward(states_, next_actions).detach()).flatten()
        # critic_value_ = critic_value_ * (1 - dones[:, 0].float())
        critic_value_[dones[:,0]] = 0.0
        target_q = rewards[:,id] + self.agents[id].gamma*critic_value_

        current_q = self.agents[id].critic.forward(states, old_actions).flatten()

        critic_loss = F.mse_loss(target_q, current_q)
        self.agents[id].critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.agents[id].critic.optimizer.step()

        # Update Actor
        actor_loss = self.agents[id].critic.forward(states, mu_actions).flatten()
        actor_loss = -T.mean(actor_loss)
        self.agents[id].actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.agents[id].actor.optimizer.step()
 
        self.agents[id].update_network_parameters()

