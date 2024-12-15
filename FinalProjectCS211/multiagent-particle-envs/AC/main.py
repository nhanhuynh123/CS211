#---------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                         #
#      Để chạy file này thì vào ./FinalProjectCS211/multiagent-particle-envs/multiagent/evironment.py và đổi self.discrete_action_input thành True        #
#                                                                                                                                                         #
#---------------------------------------------------------------------------------------------------------------------------------------------------------#

# Try not to modify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Đường dẫn thư mục hiện tại (AC)
reward_records_dir = os.path.join(current_dir, "../reward_records")  # Trỏ tới thư mục đồng cấp

file_name = "maac_reward_record.npy"
file_path = os.path.join(reward_records_dir, file_name) # Path to target folder

from maac import MultiAgentActorCritic
#from make_env import make_env
from pettingzoo.mpe import simple_adversary_v3
import numpy as np
import torch as T
if __name__ == "__main__":
    n_actions = 5
    scenario = "simple_adversary"
    #env = make_env(scenario)
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=True)
    obs = env.reset()
    
    n_agents = env.num_agents
    actor_dims = []
    env.reset()
    for agent_name in env.agents:
        actor_dims.append(env.observation_spaces[agent_name]._shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 5
    maac = MultiAgentActorCritic(actor_dims, n_actions, n_agents,n_actions, 0.001, 0.001)

    PRINT_INTERVAL = 500
    N_GAMES = 25000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    best_score = 0

    for i in range(N_GAMES):
        obs, _ = env.reset(seed=i)
        score = 0
        dones = [False] * n_agents
        episode_step = 0
        while not any(dones):
            actions = maac.choose_action(obs)
            obs_, rewards, termination, truncation, _ = env.step(actions)
            state = np.concatenate([i for i in obs.values()])
            state_ = np.concatenate([i for i in obs_.values()])            
            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents
            maac.learn(obs, actions, rewards, obs_, dones)
            score += sum(rewards.values())
        
            total_steps += 1
            episode_step += 1
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if best_score < avg_score:
            best_score = avg_score
        if i % PRINT_INTERVAL == 0:
            print("Episode ", i, " average score {:.1f}".format(avg_score))
            print("Best Score: ",best_score)
    np.save(file_path, score_history)        
    