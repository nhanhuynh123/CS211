import torch as T
import torch.nn.functional as F
from agent import Agent
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True,
                                random_drop=True, random_rotate=True, ball_mass=0.75, 
                                ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
env.reset()

class MADDPG:
    def __init__(self, n_action_p_agent, n_agents, critic_action_dims, 
                 scenario='pistonball',  alpha=0.01, beta=0.01,
                   gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_action_p_agent = n_action_p_agent
        self.critic_action_dims = critic_action_dims
        chkpt_dir += scenario
        self.agents = {}
        for agent_idx, agent_name in enumerate(env.possible_agents):
            self.agents[agent_name] = Agent(self.n_action_p_agent,
                                            self.critic_action_dims[agent_idx], self.n_agents,
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

        device = self.agents['piston_0'].actor.device

        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        obs = []
        obs_ = []
        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            obs_.append(new_states)
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)


            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            obs.append(mu_states)
            pi = agent.actor.forward(mu_states)

            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx])


        new_actions = []
        mu_actions = []
        old_actions = []
        for idx, (name, agent) in enumerate(self.agents.items()):
            if idx != 0 and idx != len(all_agents_new_actions)-1:
                new_actions.append(T.cat([acts for acts in all_agents_new_actions[idx-1, idx+2]], dim=1))
                mu_actions.append(T.cat([acts for acts in all_agents_new_mu_actions[idx-1, idx+2]], dim=1))
                old_actions.append(T.cat([acts for acts in old_agents_actions[idx-1, idx+2]], dim=1))
                
            else:
                if idx == 0:
                    new_actions.append(T.cat([acts for acts in all_agents_new_actions[idx:idx+2]], dim=1))
                    mu_actions.append(T.cat([acts for acts in all_agents_new_mu_actions[idx:idx+2]], dim=1))
                    old_actions.append(T.cat([acts for acts in old_agents_actions[idx:idx+2]], dim=1))
                else: 
                    new_actions.append(T.cat([acts for acts in all_agents_new_actions[idx-1, idx+1]], dim=1))
                    mu_actions.append(T.cat([acts for acts in all_agents_new_mu_actions[idx-1, idx+1]], dim=1))
                    old_actions.append(T.cat([acts for acts in old_agents_actions[idx-1, idx+1]], dim=1))





        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):

            critic_value_ = agent.target_critic.forward(obs_[agent_idx], new_actions[agent_idx]).flatten()

            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(obs[agent_idx], old_actions[agent_idx]).flatten()

            target = rewards[:,agent_idx].float() + agent.gamma*critic_value_


            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True, inputs=list(agent.critic.parameters()))
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(obs[agent_idx], mu_actions[agent_idx]).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True, inputs=list(agent.actor.parameters()))
            agent.actor.optimizer.step()

            agent.update_network_parameters()
