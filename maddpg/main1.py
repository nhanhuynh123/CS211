from buffer import MultiAgentReplayBuffer
from maddpg_torch import MADDPG
import numpy as np
import torch as T
from pettingzoo.mpe import simple_adversary_v3

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    scenario = 'simple_adversary'
    env = simple_adversary_v3.env(render_mode="human")
    obs = env.reset(seed=42)
    print(obs)
    if obs is None or not obs:
        raise ValueError("Environment reset returned None or empty observation.")
    
    print(f"Observation after reset: {obs}")    
    n_agents = env.num_agents
    actor_dims = []
    for name in env.agents:
        actor_dims.append(env.observation_space(name).shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space(env.agents[0]).n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        if obs is None:
            raise ValueError("Environment reset returned None. Check your environment initialization.")
        obs_list = [obs[agent] for agent in env.agents]  # Chuyển đổi obs nếu cần
        score = 0
        done = {agent: False for agent in env.agents}  # Dùng dictionary nếu môi trường trả về dạng này
        episode_step = 0

        while not any(done.values()):
            if evaluate:
                env.render()

            actions = maddpg_agents.choose_action(obs_list)  # Chọn hành động từ danh sách quan sát
            obs_, reward, done, info = env.step(actions)

            if obs_ is None:
                raise ValueError("Environment step returned None for obs_. Check your environment.")

            obs_list_ = [obs_[agent] for agent in env.agents]  # Chuyển đổi obs_ nếu cần
            reward_list = [reward[agent] for agent in env.agents]  # Chuyển đổi reward
            done_list = [done[agent] for agent in env.agents]  # Chuyển đổi done

            state = obs_list_to_state_vector(obs_list)
            state_ = obs_list_to_state_vector(obs_list_)

            if episode_step >= MAX_STEPS:
                done_list = [True] * n_agents

            memory.store_transition(obs_list, state, actions, reward_list, obs_list_, state_, done_list)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs_list = obs_list_  # Cập nhật obs
            score += sum(reward_list)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))