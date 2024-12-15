from network import Actor, Critic
import torch as T

class Agent:
    def __init__(self, state_dims, action_dims,agent_name,n_actions, gamma=0.99, alpha=0.001, beta=0.001):

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.agent_name = agent_name
        self.n_actions = n_actions
        self.actor = Actor(self.state_dims, self.action_dims, alpha)
        self.critic = Critic(self.state_dims, beta)

        self.gamma = gamma

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        min_v = (1 - actions).min()
        noise = (T.rand(self.n_actions, device=self.actor.device) * min_v).to(self.actor.device)
        action = actions + noise
        return action.detach().cpu().numpy()[0]