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
        self.actor_noise_action_memory = [] 
        self.actor_free_noise_action_memory = []
        self.actor_next_action_memory = [] 

        for i in range(self.n_agents):

            self.actor_noise_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))
            
            self.actor_free_noise_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))
            
            self.actor_next_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))
    
    def store_transition(self, state, noise_action, free_noise_action, reward
                            , state_, next_action, dones):
        index = self.mem_cntr % self.mem_size
        # print(self.actor_action_memory[0])
        for agent_idx in range(self.n_agents):
            self.actor_noise_action_memory[agent_idx][index] = noise_action[agent_idx]
            self.actor_free_noise_action_memory[agent_idx][index] = free_noise_action[agent_idx]
            self.actor_next_action_memory[agent_idx][index] = next_action[agent_idx]
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = dones

        self.mem_cntr += 1
        print("actor_noise_action_memory", self.actor_noise_action_memory)
        print("actor_free_noise_action_memory", self.actor_free_noise_action_memory)
        print("actor_next_action_memory", self.actor_next_action_memory)
        print("state_memory", self.state_memory)
        print("new_state_memory", self.new_state_memory)
        print("reward_memory", self.reward_memory)
        print("terminal_memory", self.terminal_memory)
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch] 
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        noise_actions = []
        free_noise_actions = []
        next_actions = []

        for agent_idx in range(self.n_agents):
            noise_actions.append(self.actor_noise_action_memory[agent_idx][batch])
            free_noise_actions.append(self.actor_free_noise_action_memory[agent_idx][batch])
            next_actions.append(self.actor_next_action_memory[agent_idx][batch])
        
        return states, noise_actions, free_noise_actions, rewards, \
             states_, next_actions, terminal #actor_states[n_agents, [64, 10]], states[64, critic_dims],  actions[3, [64, 1]], 
    
    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False
    
