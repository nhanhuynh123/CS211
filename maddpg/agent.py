import torch as T
from network import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01, device='cpu'):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.device = device
        
        # Tạo các mạng neural cho actor và critic
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor').to(self.device)
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic').to(self.device)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor').to(self.device)
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic').to(self.device)

        # Cập nhật các tham số mạng target
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, noise_factor=0.1):
        # Chuyển đổi observation thành tensor và đưa vào device
        state = T.tensor([observation], dtype=T.float).to(self.device)
        
        # Tính hành động từ mạng actor
        actions = self.actor.forward(state)
        
        # Thêm nhiễu vào hành động cho mục đích khám phá
        noise = noise_factor * T.randn_like(actions).to(self.device)  # Thêm nhiễu Gaussian
        action = actions + noise
        
        # Trả về hành động đã được làm tròn (detached) và chuyển về numpy để môi trường sử dụng
        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Cập nhật tham số của target_actor và target_critic
        self._update_target_network(self.actor, self.target_actor, tau)
        self._update_target_network(self.critic, self.target_critic, tau)

    def _update_target_network(self, network, target_network, tau):
        # Cập nhật các tham số của mạng target
        network_params = network.named_parameters()
        target_network_params = target_network.named_parameters()

        network_state_dict = dict(network_params)
        target_network_state_dict = dict(target_network_params)

        for name in network_state_dict:
            network_state_dict[name] = tau * network_state_dict[name].clone() + \
                                       (1 - tau) * target_network_state_dict[name].clone()

        target_network.load_state_dict(network_state_dict)

    def save_models(self):
        # Lưu các mô hình của agent
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        # Tải các mô hình của agent
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
