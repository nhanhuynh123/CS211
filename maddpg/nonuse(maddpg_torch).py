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
    
    def choose_action(self, obs, agent_index):
        
        actions = self.agents[agent_index].choose_action(obs)
        # print(actions)
        noise_actions = np.clip(actions, 0.0, 1.0)
        return noise_actions
    

    
    def learn(self, memory):
        if not memory.ready():
            return
        
        actor_states, states, noise_actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer() 

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        noise_actions = T.tensor(noise_actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        # Get old action, new action
        all_agents_old_actions = []

        for agent_idx, agent in enumerate(self.agents):
            all_agents_old_actions.append(noise_actions[agent_idx])
        old_actions = T.cat([act for act in all_agents_old_actions], dim = 1)

        # Get mu action
        new_actions = []
        mu = []

        for i in range(self.n_agents):
            all_agents_mu_actions = []
            all_agents_new_actions = []
            for agent_idx, agent in enumerate(self.agents):

                new_state = T.tensor(actor_new_states[agent_idx], 
                                dtype=T.float).to(device)
                new_pi = agent.target_actor.forward(new_state)

                state = T.tensor(actor_states[agent_idx],
                            dtype=T.float).to(device)
                pi = agent.actor.forward(state)

                if agent_idx == i:
                    all_agents_mu_actions.append(pi)
                    all_agents_new_actions.append(new_pi)
                else: 
                    all_agents_mu_actions.append(pi.detach())
                    all_agents_new_actions.append(new_pi.detach())

            new_actions.append(T.cat([act for act in all_agents_new_actions], dim = 1))
            mu.append(T.cat([act for act in all_agents_mu_actions], dim = 1))

        # Update critic
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions[agent_idx]).flatten()
            critic_value_[dones[:,0]] = 0.0

            target = rewards[:,0] + agent.gamma*critic_value_

            current_q = agent.critic.forward(states, old_actions.detach()).flatten()
            critic_loss = F.mse_loss(target, current_q)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # Update Actor
            actor_loss = agent.critic.forward(states, mu[agent_idx]).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
    
            agent.update_network_parameters()

# mocdel.eval() để kiểm tra mô hình