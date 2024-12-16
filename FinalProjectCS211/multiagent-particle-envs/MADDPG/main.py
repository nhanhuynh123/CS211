#---------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                         #
#      Để chạy file này thì vào ./FinalProjectCS211/multiagent-particle-envs/multiagent/evironment.py và đổi self.discrete_action_input thành Fasle       #
#                                                                                                                                                         #
#---------------------------------------------------------------------------------------------------------------------------------------------------------#


# Try not to modify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Đường dẫn thư mục hiện tại (AC)
reward_records_dir = os.path.join(current_dir, "../reward_records")  # Trỏ tới thư mục đồng cấp

file_name = "maddpg_reward_record.npy"
file_path = os.path.join(reward_records_dir, file_name) # Path to target folder

import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3
import warnings
import time

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    # scenario = 'simple'
    scenario = 'simple_adversary'
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=True)
    obs = env.reset()

    n_agents = env.num_agents
    actor_dims = []
    for agent_name in env.agents:
        actor_dims.append(env.observation_spaces[agent_name]._shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 5
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=64, fc2=64,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024,
                                    agent_names=env.agents)

    PRINT_INTERVAL = 500
    N_GAMES = 25000
    MAX_STEPS =25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = -3
    #seeds = np.load("/Users/cicilian/Desktop/Multi-Agent-Deep-Deterministic-Policy-Gradients-PyTorch/seed.npy")
    if evaluate:
        maddpg_agents.load_checkpoint()
    time.sleep(1)
    for i in range(N_GAMES):
        obs, _ = env.reset(seed=i)
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()

                time.sleep(0.12) # to slow down the action for the video

            actions = maddpg_agents.choose_action(obs)
            obs_, reward, termination, truncation, _ = env.step(actions)
            state = np.concatenate([i for i in obs.values()])
            state_ = np.concatenate([i for i in obs_.values()])

            if episode_step >= MAX_STEPS:
                done = [True] * n_agents

            if any(termination.values()) or any(truncation.values()) or (episode_step >= MAX_STEPS):
                done = [True] * n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward.values())
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:

            if (avg_score > best_score) and (i > PRINT_INTERVAL):
                print(" avg_score, best_score", avg_score, best_score)
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
    np.save(file_path, score_history)