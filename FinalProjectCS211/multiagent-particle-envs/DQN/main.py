
# Try not to modify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from save_file import save_with_unique_name


import numpy as np
from madqn import MultiAgentDQN
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    n_actions = 5  # Số lượng hành động
    scenario = "simple_adversary"
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=False)
    env.reset()
    n_agents = env.num_agents
    actor_dims = []
    for agent_name in env.agents:
        actor_dims.append(env.observation_spaces[agent_name]._shape[0])

    # Sử dụng seed để tái lập kết quả
    # SEED = 42
    # np.random.seed(SEED)
    # env.reset(seed=SEED)
    
    # Thông số của agent và environment
    madqn = MultiAgentDQN(actor_dims, n_actions, gamma=0.99, lr=0.001, epsilon=0.1)

    # Khởi tạo buffer
    replay_buffer = MultiAgentReplayBuffer(
        1000000,
        actor_dims=actor_dims,
        n_actions=1,
        n_agents=n_agents,
        batch_size=1024,
        agent_names=env.agents
    )

    obs = env.reset()

    # Thông số huấn luyện
    PRINT_INTERVAL=500
    N_GAMES = 100000
    EXPLORATION_FRACTION = 0.5
    MAX_STEPS = 25
    TRAIN_FREQUENCY = 10
    TARGET_UPDATE_FREQUENCY = 500
    total_steps = 0
    score_history = []
    best_score = -np.inf

    for i in range(N_GAMES):
        # Reset lại môi trường
        obs, _ = env.reset(seed=i)
        score = 0
        done = [False] * n_agents
        episode_step = 0

        while not any(done):
            # Chọn hành động từ các agent
            epsilon = linear_schedule(1, 0.001,N_GAMES * EXPLORATION_FRACTION, i)
            actions = madqn.choose_actions(obs, epsilon)
            # print(actions)
            # Thực hiện hành động
            next_obs, rewards, termination, truncation, _ = env.step(actions)

            if any(termination.values()) or any(truncation.values()) or (episode_step >= MAX_STEPS):
                done = [True] * n_agents

            # Lưu vào buffer
            replay_buffer.store_transition(
                raw_obs=obs,
                action=actions,
                reward=rewards,
                raw_obs_=next_obs,
                done=done
            )
            if replay_buffer.ready():
                if total_steps % TRAIN_FREQUENCY == 0:
                    madqn.train(replay_buffer)
                if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                    madqn.update_target_network()
                
            obs = next_obs
            score += sum(rewards.values())
            total_steps += 1
            episode_step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        # Lưu checkpoint khi có kết quả tốt nhất
        if avg_score > best_score:
            best_score = avg_score
        if i % PRINT_INTERVAL == 0:
            print("Episode ", i, " average score {:.1f}".format(avg_score))
            print("Best Score: ",best_score)

    file_name = "madqn_reward_record"
    save_with_unique_name(file_name=file_name, data=score_history)