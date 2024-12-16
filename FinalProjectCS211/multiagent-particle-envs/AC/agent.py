from network import Actor, Critic
import torch as T
import numpy as np

class Agent:
    def __init__(self, state_dims, action_dims,agent_name, gamma=0.99, alpha=0.001, beta=0.001):

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.agent_name = agent_name
        self.actor = Actor(self.state_dims, self.action_dims, alpha)
        self.critic = Critic(self.state_dims, beta)

        self.gamma = gamma

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions_probs = self.actor.forward(state).detach().numpy()[0]
        actions = np.random.choice(len(actions_probs), p=actions_probs)
        return actions