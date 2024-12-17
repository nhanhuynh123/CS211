# # ddpg_multiagent.py

# import os
# import torch as T
# import numpy as np
# import torch.nn.functional as F
# from network import ActorNetwork, CriticNetwork
# from buffer import ReplayBuffer

# class DDPGMultiAgent:
#     def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
#                  scenario='simple_adversary', alpha=0.001, beta=0.001, fc1=400, fc2=300,
#                  gamma=0.99, tau=0.005, chkpt_dir='tmp/ddpg_multiagent/'):
#         self.agents = []
#         self.n_agents = n_agents
#         self.n_actions = n_actions
#         chkpt_dir += f'_{scenario}'
#         self.gamma = gamma
#         self.tau = tau
#         self.memory = [ReplayBuffer(1000000, actor_dims[agent_idx], n_actions, batch_size=1024) for agent_idx in range(self.n_agents)]
        
#         self.noise_scale = 0.1  # Có thể điều chỉnh

#         for agent_idx in range(self.n_agents):
#             agent = {
#                 'actor': ActorNetwork(alpha, actor_dims[agent_idx], fc1, fc2,
#                                       n_actions, name=f'actor_{agent_idx}', chkpt_dir=chkpt_dir),
#                 'critic': CriticNetwork(beta, critic_dims[agent_idx], fc1, fc2,
#                                         n_actions, name=f'critic_{agent_idx}', chkpt_dir=chkpt_dir),
#                 'target_actor': ActorNetwork(alpha, actor_dims[agent_idx], fc1, fc2,
#                                             n_actions, name=f'target_actor_{agent_idx}', chkpt_dir=chkpt_dir),
#                 'target_critic': CriticNetwork(beta, critic_dims[agent_idx], fc1, fc2,
#                                               n_actions, name=f'target_critic_{agent_idx}', chkpt_dir=chkpt_dir),
#             }
#             # Khởi tạo target networks giống mạng chính
#             agent['target_actor'].load_state_dict(agent['actor'].state_dict())
#             agent['target_critic'].load_state_dict(agent['critic'].state_dict())
#             self.agents.append(agent)

#     def save_checkpoint(self):
#         print('... saving checkpoint ...')
#         for agent in self.agents:
#             agent['actor'].save_checkpoint()
#             agent['critic'].save_checkpoint()
#             agent['target_actor'].save_checkpoint()
#             agent['target_critic'].save_checkpoint()

#     def load_checkpoint(self):
#         print('... loading checkpoint ...')
#         for agent in self.agents:
#             agent['actor'].load_checkpoint()
#             agent['critic'].load_checkpoint()
#             agent['target_actor'].load_checkpoint()
#             agent['target_critic'].load_checkpoint()

#     def choose_noise_action(self, obs, agent_index):
#         actor = self.agents[agent_index]['actor']
#         actor.eval()
#         state = T.tensor(obs, dtype=T.float).to(actor.device)
#         mu = actor(state).cpu().detach().numpy()
#         actor.train()
#         noise = np.random.normal(scale=self.noise_scale, size=self.n_actions)
#         mu += noise
#         return np.clip(mu, -1, 1)  # Giới hạn hành động trong khoảng [-1, 1]

#     def choose_unnoise_action(self, obs, agent_index):
#         actor = self.agents[agent_index]['actor']
#         actor.eval()
#         state = T.tensor(obs, dtype=T.float).to(actor.device)
#         mu = actor(state).cpu().detach().numpy()
#         actor.train()
#         return mu

#     def update_network_parameters(self, agent_index, tau=None):
#         if tau is None:
#             tau = self.tau

#         # Cập nhật Actor
#         actor_params = self.agents[agent_index]['actor'].named_parameters()
#         target_actor_params = self.agents[agent_index]['target_actor'].named_parameters()

#         actor_state_dict = dict(actor_params)
#         target_actor_state_dict = dict(target_actor_params)

#         for name in actor_state_dict:
#             actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

#         self.agents[agent_index]['target_actor'].load_state_dict(actor_state_dict)

#         # Cập nhật Critic
#         critic_params = self.agents[agent_index]['critic'].named_parameters()
#         target_critic_params = self.agents[agent_index]['target_critic'].named_parameters()

#         critic_state_dict = dict(critic_params)
#         target_critic_state_dict = dict(target_critic_params)

#         for name in critic_state_dict:
#             critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()

#         self.agents[agent_index]['target_critic'].load_state_dict(critic_state_dict)

#     def learn(self, states, actions, rewards, next_states, dones, agent_index):
#         agent = self.agents[agent_index]
#         buffer = self.memory[agent_index]
#         buffer.store_transition(states, actions, rewards, next_states, dones)

#         if buffer.ready():
#             states_b, actions_b, rewards_b, states__b, dones_b = buffer.sample_buffer()

#             # Chuyển đổi sang tensor
#             states_b = T.tensor(states_b, dtype=T.float).to(agent['critic'].device)
#             actions_b = T.tensor(actions_b, dtype=T.float).to(agent['critic'].device)
#             rewards_b = T.tensor(rewards_b, dtype=T.float).to(agent['critic'].device)
#             states__b = T.tensor(states__b, dtype=T.float).to(agent['critic'].device)
#             dones_b = T.tensor(dones_b, dtype=T.float).to(agent['critic'].device)

#             # ----------------------- Cập Nhật Critic ----------------------- #
#             target_actions = agent['target_actor'](states__b)
#             critic_value_ = agent['target_critic'](states__b, target_actions)
#             critic_value = agent['critic'](states_b, actions_b)

#             target = rewards_b + self.gamma * critic_value_ * (1 - dones_b)
#             critic_loss = F.mse_loss(critic_value, target.detach())

#             agent['critic'].optimizer.zero_grad()
#             critic_loss.backward()
#             agent['critic'].optimizer.step()

#             # ----------------------- Cập Nhật Actor ----------------------- #
#             actor_loss = -agent['critic'](states_b, agent['actor'](states_b)).mean()

#             agent['actor'].optimizer.zero_grad()
#             actor_loss.backward()
#             agent['actor'].optimizer.step()

#             # ----------------------- Cập Nhật Mạng Mục Tiêu ----------------------- #
#             self.update_network_parameters(agent_index)


import torch as T
import torch.nn.functional as F
from agent import Agent
from pettingzoo.mpe import simple_adversary_v3
env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
env.reset()

class DDPG:
    def __init__(self, actor_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/ddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.agents = {}
        for agent_idx, agent_name in enumerate(env.possible_agents):
            self.agents[agent_name] = Agent(actor_dims[agent_idx],
                                            n_actions, n_agents,
                                            agent_name = agent_name,
                                            alpha=alpha,
                                            beta=beta,
                                            chkpt_dir=chkpt_dir)


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.load_models()

    # def choose_action(self, raw_obs):
    #     actions = []
    #     for agent_idx, agent in enumerate(self.agents):
    #         action = agent.choose_action(raw_obs[agent_idx])
    #         actions.append(action)
    #     return actions

    def choose_action(self, raw_obs):
        actions = {agent.agent_name: agent.choose_action(raw_obs[agent.agent_name]) for agent in self.agents.values()}
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, actions, rewards, \
        actor_new_states, dones = memory.sample_buffer()

        device = self.agents['adversary_0'].actor.device

        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)


            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)

            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx])




        # new_actions = T.tensor(all_agents_new_actions).to(device)
        # mu = T.tensor(all_agents_new_mu_actions).to(device)
        # old_actions = T.tensor(old_agents_actions).to(device)



        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):
            states_ = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            old_actions = old_agents_actions[agent_idx]
            # print(old_actions)
            # print(all_agents_new_actions[agent_idx])
            critic_value_ = agent.target_critic.forward(states_, all_agents_new_actions[agent_idx]).flatten()

            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx].float() + agent.gamma*critic_value_


            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True, inputs=list(agent.critic.parameters()))
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, all_agents_new_mu_actions[agent_idx]).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True, inputs=list(agent.actor.parameters()))
            agent.actor.optimizer.step()

            agent.update_network_parameters()
