
# Try not to modify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from save_file import save_with_unique_name
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from dataclasses import dataclass
import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3
import warnings
import time
import tyro 

@dataclass
class Args:
    epochs: int  = 25000
    # Number games
    PRINT_INTERVAL: int = 1
    # Print frequency
    MAX_STEPS: int = 25
    # Max episode steps
    demo: bool = False

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    args = tyro.cli(Args)
    dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "tmp/maddpg")
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
    #os.makedirs("tmp/maddpg", exist_ok=True)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=64, fc2=64,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir=dir)

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024,
                                    agent_names=env.agents)

    total_steps = 0
    score_history = []
    
    best_score = -float("inf")
    if args.demo:
        maddpg_agents.load_checkpoint()
    time.sleep(1)
    for i in range(args.epochs):
        obs, _ = env.reset(seed=i)
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):

            if args.demo:  # Điều kiện args.demo
                frame = env.render()
                time.sleep(0.5)  # Điều chỉnh tốc độ cho video

            actions = maddpg_agents.choose_action(obs)
            obs_, reward, termination, truncation, _ = env.step(actions)
            state = np.concatenate([i for i in obs.values()])
            state_ = np.concatenate([i for i in obs_.values()])

            if episode_step >= args.MAX_STEPS:
                done = [True] * n_agents

            if any(termination.values()) or any(truncation.values()) or (episode_step >= args.MAX_STEPS):
                done = [True] * n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            
            if total_steps % 100 == 0 and not args.demo:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward.values())
            total_steps += 1
            episode_step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not args.demo:

            if (avg_score > best_score):
                best_score = avg_score
        if i % args.PRINT_INTERVAL == 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            print("best_score", best_score)
    file_name = "maddpg_reward_record" + f"_{args.epochs}"
    save_with_unique_name(file_name=file_name, data=score_history)
