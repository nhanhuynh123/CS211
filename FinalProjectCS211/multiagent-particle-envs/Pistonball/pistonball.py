# Try not to modify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from save_file import save_with_unique_name

from dataclasses import dataclass
import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.butterfly import pistonball_v6
import warnings
import time
import tyro 
import supersuit as ss
@dataclass

class Args:
    epochs: int  = 25000
    # Number games
    PRINT_INTERVAL: int = 500
    # Print frequency
    MAX_STEPS: int = 25
    # Max episode steps
    demo = False

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    # scenario = 'simple'
    args = tyro.cli(Args) 
    scenario = 'pistonball_v6'
    env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True,
                                random_drop=True, random_rotate=True, ball_mass=0.75, 
                                ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)

    env = ss.color_reduction_v0(env, mode='G')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    obs, _ = env.reset()
    n_agents = len(env.possible_agents)
    # Action dims of critc network
    critic_action_dims = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
    # number action of each agent
    n_action_p_agent = env.action_space("piston_0").shape[0]
    # Number dimension of observation
    actor_dims = obs["piston_0"].shape


    # action space is a list of arrays, assume each agent has same action space
    maddpg_agents = MADDPG(n_action_p_agent, n_agents, critic_action_dims,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000, actor_dims,
                                    n_action_p_agent, n_agents, batch_size=32,
                                    agent_names=env.possible_agents)

    total_steps = 0
    score_history = []
    best_score = -float("inf")

    if args.demo:
        maddpg_agents.load_checkpoint()
    time.sleep(1)
    for i in range(args.epochs):
        print("Episode " , i)
        obs, _ = env.reset()
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if args.demo:
                env.render()

                time.sleep(0.12) # to slow down the action for the video


            actions = maddpg_agents.choose_action(obs)

            obs_, reward, termination, truncation, _ = env.step(actions)


            if episode_step >= args.MAX_STEPS:
                done = [True] * n_agents

            if any(termination.values()) or any(truncation.values()) or (episode_step >= args.MAX_STEPS):
                done = [True] * n_agents

            memory.store_transition(obs, actions, reward, obs_, done)

            if total_steps % 100 == 0 and not args.demo:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward.values())
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not args.demo:

            if (avg_score > best_score) and (i > args.PRINT_INTERVAL):
                print(f"average score: {avg_score}, best score: {best_score}")
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % args.PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))

    file_name = "piston_ball_v6"
    save_with_unique_name(file_name=file_name, data=score_history)




# from pettingzoo.butterfly import pistonball_v6
# import supersuit as ss
# import numpy as np
# env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True,
#                                 random_drop=True, random_rotate=True, ball_mass=0.75, 
#                                 ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
# obs, _ = env.reset()

# env = ss.color_reduction_v0(env, mode='G')
# env = ss.resize_v1(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

# # n_agents = env.num_agents
# print(len(env.possible_agents))
# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     # print(actions)
#     observations, rewards, terminations, truncations, infos = env.step(actions)

#     # name = "piston_18"
#     # np.save("/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/MADDPG/_.npy",observations[name][:, :, 0])
#     # np.save("/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/MADDPG/__.npy",observations[name][:, :, 1])
#     # np.save("/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/MADDPG/___.npy",observations[name][:, :, 2])
#     i += 1
#     print(i)
#     if i == 2:
#         break
# env.close()