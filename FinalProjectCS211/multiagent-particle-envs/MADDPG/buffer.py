import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0  #memori counter
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.actor_dims = actor_dims

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)

        self.init_actor_memory()
    
    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_noise_action_memory = []

        for i in range(self.n_agents):

            self.actor_noise_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
    
    def store_transition(self, obs, state, noise_action, reward, obs_
                            , state_, dones):
        index = self.mem_cntr % self.mem_size
        # print(self.actor_action_memory[0])
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = obs_[agent_idx]
            self.actor_noise_action_memory[agent_idx][index] = noise_action[agent_idx]
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = dones

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch] 
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        noise_actions = []
        actor_states = []
        actor_new_states = []
        

        for agent_idx in range(self.n_agents):
            noise_actions.append(self.actor_noise_action_memory[agent_idx][batch])
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
        
        return actor_states, states, noise_actions, rewards, \
             actor_new_states, states_, terminal #actor_states[n_agents, [64, 10]], states[64, critic_dims],  actions[3, [64, 1]], 
    
    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False
    
