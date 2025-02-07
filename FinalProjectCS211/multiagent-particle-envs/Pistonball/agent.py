import torch as T
from network import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, n_action_p_agent, critic_action_dims, n_agents,  chkpt_dir, agent_name,
                    alpha=0.01, beta=0.01, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.critic_action_dims = critic_action_dims
        self.agent_name = agent_name
        self.actor = ActorNetwork(alpha, n_action_p_agent, chkpt_dir=chkpt_dir,  
                                  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_action_dims, chkpt_dir=chkpt_dir, 
                                    name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, n_action_p_agent, chkpt_dir=chkpt_dir,
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_action_dims, chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state)
        min_v = (1-action)
        noise = (T.rand(1).to(self.actor.device) * min_v).to(self.actor.device)

        action_ = action + noise
        return action_.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
