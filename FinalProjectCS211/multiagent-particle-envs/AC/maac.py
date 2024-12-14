import torch as T
import numpy as np
from agent import Agent

class MultiAgentActorCritic:
    def __init__(self, actor_dims, action_dims, num_agents, gamma=0.99, alpha=0.001, beta=0.001):
        self.agents = []
        self.num_agents = num_agents
        for agent_idx in range(self.num_agents):
            self.agents.append(Agent(actor_dims[agent_idx], action_dims, gamma, alpha, beta))
    
    def choose_action(self, obs):
        actions_probs = []
        with T.no_grad():
            for idx, agent in enumerate(self.agents):
                actions_probs.append(agent.choose_action(obs[idx]))

        actions = []
        for i in range(len(actions_probs)):
            actions.append(np.random.choice(actions_probs[i].shape[0]))
        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        device = self.agents[0].actor.device
        for idx, agent in enumerate(self.agents):

            state = T.tensor(states[idx], dtype=T.float).unsqueeze(0).to(device)
            next_state = T.tensor(next_states[idx], dtype=T.float).to(device)
            reward = T.tensor(rewards[idx], dtype=T.float).to(device)
            done = T.tensor(dones[idx]).to(device)
            action_probs = agent.actor.forward(state)
            
            # Calculate V(s_t) and V(s_t+1)
            
            state_value = agent.critic.forward(state)[0]
            next_state_value = agent.critic.forward(next_state)[0]
            # Calculate advantage
            next_state_value[done] = 0.0
            advantage = reward + agent.gamma * next_state_value - state_value

            # Calculate actor_loss, critic_loss
            log_prob = T.log(action_probs[0, actions[idx]])
            actor_loss = -log_prob * advantage.detach()
            print(advantage)
            critic_loss = advantage**2
            # print(log_prob)
            # print(advantage)
            # Update actor
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()

            # Update critic
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()