from agent import DQNAgent
import torch as T
from pettingzoo.mpe import simple_adversary_v3

env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
env.reset()
class MultiAgentDQN:
    def __init__(self, obs_spaces, action_spaces, gamma=0.99, lr=0.001, epsilon=0.1):
        self.agents = {}
        for agent_idx, agent_name in enumerate(env.possible_agents):
        # for obs, act in zip(obs_spaces, action_spaces):
            # self.agents.append(DQNAgent(obs.shape[0], act.n, gamma, lr, epsilon))
            self.agents[agent_name] = DQNAgent(obs_spaces[agent_idx], action_spaces, agent_name, gamma, lr, epsilon)

    def choose_actions(self, observations, epsilon):
        actions = {agent.agent_name: agent.choose_action (observations[agent.agent_name], epsilon) for agent in self.agents.values()}
        return actions

    def train(self, memory):
        device = self.agents['adversary_0'].q_network.device
        states, actions, rewards, next_states, dones = memory.sample_buffer()
        
        actions = T.tensor(actions, dtype=T.int64).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        # print(len(states[0]))
        # print(actions[0].shape)
        # print(rewards[:, 0].shape)
        # print(dones[:, 0].shape)

        for agent_idx, agent in enumerate(self.agents.values()):

            state = states[agent_idx]
            action = actions[agent_idx]
            reward = rewards[:, agent_idx]
            next_state = next_states[agent_idx]
            done = dones[:, agent_idx]

            self.agents[agent.agent_name].learn(state, action, reward, next_state, done)
    
    def update_target_network(self):
        for idx, agent in enumerate(self.agents.values()):
            self.agents[agent.agent_name].update_target_network()