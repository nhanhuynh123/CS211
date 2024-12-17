# from ddpg_multiagent import DDPGMultiAgent
# import numpy as np
# import torch as T
# from pettingzoo.mpe import simple_adversary_v3

# def obs_to_state(observation):
#     return np.array(observation, dtype=np.float32)

# if __name__ == '__main__':
#     scenario = "simple_adversary"
#     env = simple_adversary_v3.env(continuous_actions=True)
#     env.reset()
#     n_agents = env.num_agents
#     actor_dims = []
#     for name in env.agents:
#         actor_dims.append(env.observation_space(name).shape[0])
#     critic_dims = actor_dims  # Trong DDPG, Critic chỉ nhận state của chính mình

#     n_actions = env.action_space(env.agents[0]).shape[0]

#     ddpg_agents = DDPGMultiAgent(actor_dims, critic_dims, n_agents, n_actions,
#                                  scenario=scenario,
#                                  alpha=0.001, beta=0.001,
#                                  fc1=400, fc2=300,
#                                  gamma=0.99, tau=0.005,
#                                  chkpt_dir='tmp/ddpg_multiagent/')
    
#     PRINT_INTERVAL = 500
#     N_GAMES = 1000
#     args. = 25
#     total_steps = 0
#     score_history = []
#     evaluate = False
#     best_score = -np.inf

#     if evaluate:
#         ddpg_agents.load_checkpoint()

#     for i in range(N_GAMES):
#         env.reset()
#         score = 0
#         dones = [False] * n_agents
#         episode_step = 0
#         obs = []
#         for name in env.agents:
#             obs.append(env.observe(name))

#         while not any(dones):
#             if evaluate:
#                 env.render()

#             obs_ = []
#             rewards = []
#             dones_temp = []
#             noise_actions = []
#             unnoise_actions = []

#             for id, name in enumerate(env.agents):
#                 noise_action = ddpg_agents.choose_noise_action(obs[id], id)
#                 unnoise_action = ddpg_agents.choose_unnoise_action(obs[id], id)
#                 noise_actions.append(noise_action)
#                 unnoise_actions.append(unnoise_action)

#             actions = noise_actions  # Sử dụng hành động có nhiễu để khám phá

#             # Thực hiện các hành động trong môi trường
#             env.step(actions)

#             for id, name in enumerate(env.agents):
#                 obs_.append(env.observe(name))
#                 rewards.append(env.rewards[name])
#                 dones_temp.append(env.terminations[name] or env.truncations[name])

#             for id in range(n_agents):
#                 # Lưu trữ chuyển tiếp vào bộ nhớ của từng agent
#                 ddpg_agents.learn(obs[id], actions[id], rewards[id], obs_[id], dones_temp[id], id)

#             obs = obs_

#             score += sum(rewards)
#             total_steps += 1
#             episode_step += 1

#             if episode_step > MAX_STEPS:
#                 dones = [True] * n_agents

#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])
#         if not evaluate:
#             if avg_score > best_score:
#                 ddpg_agents.save_checkpoint()
#                 best_score = avg_score
#         if i % PRINT_INTERVAL == 0 and i > 0:
#             print('Episode:', i, 'Average Score {:.1f}'.format(avg_score))


# Try not to modify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from save_file import save_with_unique_name


import numpy as np
from ddpg import DDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3
import warnings
import time
from dataclasses import dataclass
import tyro

@dataclass
class Args:
    epochs: int  = 25000
    # Number games
    PRINT_INTERVAL: int = 500
    # Print frequency
    MAX_STEPS: int = 25
    # Max episode steps

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    # scenario = 'simple'
    args = tyro.cli(Args)
    scenario = 'simple_adversary'
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=True)
    obs = env.reset()

    n_agents = env.num_agents
    actor_dims = []
    for agent_name in env.agents:
        actor_dims.append(env.observation_spaces[agent_name]._shape[0])

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 5
    ddpg_agents = DDPG(actor_dims, n_agents, n_actions,
                           fc1=64, fc2=64,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, actor_dims,
                                    n_actions, n_agents, batch_size=1024,
                                    agent_names=env.agents)


    total_steps = 0
    score_history = []
    evaluate = False
    best_score = -3
    if evaluate:
        ddpg_agents.load_checkpoint()
    time.sleep(1)
    for i in range(args.epochs):
        obs, _ = env.reset(seed=i)
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()

                time.sleep(0.12) # to slow down the action for the video

            actions = ddpg_agents.choose_action(obs)
            obs_, reward, termination, truncation, _ = env.step(actions)

            if episode_step >= args.epochs:
                done = [True] * n_agents

            if any(termination.values()) or any(truncation.values()) or (episode_step >= args.MAX_STEPS):
                done = [True] * n_agents

            memory.store_transition(obs, actions, reward, obs_, done)
            
            if total_steps % 100 == 0 and not evaluate:
                ddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward.values())
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:

            if (avg_score > best_score):
                best_score = avg_score
        if i % args.PRINT_INTERVAL == 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            print("best_score", best_score)

    file_name = "ddpg_reward_record"
    save_with_unique_name(file_name=file_name, data=score_history)