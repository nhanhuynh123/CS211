import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, actor_dims, 
            n_actions, n_agents, batch_size,agent_names):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.agent_names = agent_names

        # self.state_memory = np.zeros((self.mem_size, critic_dims))
        # self.new_state_memory = np.zeros((self.mem_size, critic_dims  ))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))


    def store_transition(self, raw_obs, action, reward, 
                               raw_obs_, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
        
        index = self.mem_cntr % self.mem_size

        for agent_idx, agent_name in enumerate(self.agent_names):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_name]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_name]
            self.actor_action_memory[agent_idx][index] = action[agent_name]

        self.reward_memory[index] = [i for i in reward.values()]
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, actions, rewards, \
               actor_new_states, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
