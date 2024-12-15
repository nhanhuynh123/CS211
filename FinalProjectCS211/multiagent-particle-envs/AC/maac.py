import torch as T
import numpy as np
from agent import Agent
from pettingzoo.mpe import simple_adversary_v3
env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
env.reset()
class MultiAgentActorCritic:
    def __init__(self, actor_dims, action_dims, num_agents, gamma=0.99, alpha=0.001, beta=0.001):
        self.agents = []
        self.num_agents = num_agents
        self.agents = {}
        for agent_idx, agent_name in enumerate(env.possible_agents):
            self.agents[agent_name]= Agent(actor_dims[agent_idx], action_dims,agent_name, gamma, alpha, beta)

    
    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def choose_action(self, raw_obs):
        
        actions = {agent.agent_name: agent.choose_action(raw_obs[agent.agent_name]) for agent in self.agents.values()}
        return actions

    def learn(self, obs, actions, rewards, next_obs, dones):
        device = self.agents['adversary_0'].actor.device

        # Chuyển đổi obs và next_obs thành danh sách các trạng thái
        states = [obs[agent_name] for agent_name in self.agents.keys()]
        next_states = [next_obs[agent_name] for agent_name in self.agents.keys()]
        
        # Chuyển đổi rewards và dones thành danh sách
        rewards_list = [rewards[agent_name] for agent_name in self.agents.keys()]
        dones_list = [dones[idx] for idx, _ in enumerate(self.agents.keys())]


        for idx, (agent_name, agent) in enumerate(self.agents.items()):
            # Xử lý trạng thái và phần thưởng cho từng agent
            state = T.tensor(states[idx], dtype=T.float).unsqueeze(0).to(device)
            next_state = T.tensor(next_states[idx], dtype=T.float).to(device)
            reward = T.tensor(rewards_list[idx], dtype=T.float).to(device)
            done = T.tensor(dones_list[idx]).to(device)
            # Hành động của agent hiện tại
            action_probs = agent.actor.forward(state)
            
            # Tính giá trị trạng thái
            state_value = agent.critic.forward(state)[0]
            next_state_value = agent.critic.forward(next_state)[0]

            # Tính advantage
            next_state_value[done] = 0.0
            advantage = reward + agent.gamma * next_state_value - state_value

            # Tính toán actor_loss và critic_loss
            log_prob = T.log(action_probs[0, actions[agent_name]])
            
            #actor_loss = -log_prob * advantage.detach()
            actor_loss = (-log_prob * advantage.detach()).mean()
            critic_loss = advantage ** 2

            # Cập nhật actor
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()

            # Cập nhật critic
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()
