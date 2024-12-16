import sys
import os
import numpy as np
import time
from agent import DQNAgent
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3
import matplotlib.pyplot as plt

# Thêm đường dẫn cho module khác nếu cần
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def obs_list_to_state_vector(observation):
    """Chuyển đổi danh sách quan sát thành vector trạng thái."""
    state = np.array([])
    for obs in observation.values():
        state = np.concatenate([state, obs])
    return state

if __name__ == "__main__":
    # Thiết lập môi trường PettingZoo
    scenario = "simple_adversary"
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=False)
    
    # Sử dụng seed để tái lập kết quả
    SEED = 42
    np.random.seed(SEED)
    env.reset(seed=SEED)
    
    # Thông số của agent và environment
    agents = env.agents
    n_agents = env.num_agents
    input_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}  # Sử dụng observation_space
    n_actions = env.action_spaces[agents[0]].n  # Số lượng hành động
    critic_dims = sum([env.observation_space(agent).shape[0] for agent in agents])  # Sử dụng observation_space

    # Khởi tạo DQN cho từng agent
    dqn_agents = {agent: DQNAgent(input_dims[agent], n_actions, lr=0.001, gamma=0.99, epsilon=1.0) for agent in agents}

    # Khởi tạo buffer
    buffer_size = 100000
    batch_size = 64
    replay_buffer = MultiAgentReplayBuffer(
        max_size=buffer_size,
        critic_dims=critic_dims,
        actor_dims=[input_dims[agent] for agent in agents],
        n_actions=n_actions,
        n_agents=n_agents,
        batch_size=batch_size,
        agent_names=agents
    )

    # Thông số huấn luyện
    N_GAMES = 25000
    MAX_STEPS = 25
    TARGET_UPDATE_INTERVAL = 10
    total_steps = 0
    score_history = []
    best_score = -np.inf

    for i in range(N_GAMES):
        # Reset lại môi trường
        obs, _ = env.reset(seed=i)
        score = 0
        done = {agent: False for agent in agents}
        episode_step = 0

        while not all(done.values()):
            # Chọn hành động từ các agent
            actions = {agent: dqn_agents[agent].choose_action(obs[agent]) for agent in agents}

            # Thực hiện hành động
            next_obs, rewards, termination, truncation, _ = env.step(actions)
            if episode_step >= MAX_STEPS:
                done = {agent: True for agent in agents}
            state = obs_list_to_state_vector(obs)
            next_state = obs_list_to_state_vector(next_obs)

            # Lưu vào buffer
            replay_buffer.store_transition(
                raw_obs=obs,
                state=state,
                action=actions,
                reward=rewards,
                raw_obs_=next_obs,
                state_=next_state,
                done={agent: termination[agent] or truncation[agent] for agent in agents}
            )

            obs = next_obs
            score += sum(rewards.values())
            total_steps += 1
            episode_step += 1

            # Cập nhật target network
            if total_steps % TARGET_UPDATE_INTERVAL == 0:
                for agent in agents:
                    dqn_agents[agent].update_target_network()

            # Huấn luyện agent từ buffer
            if replay_buffer.ready():
                actor_states, states, actions, rewards_, actor_new_states, states_, terminal = replay_buffer.sample_buffer()
    for idx, agent in enumerate(agents):
        dqn_agents[agent].learn(
            state=actor_states[idx],
            action=np.argmax(actions[idx], axis=1)[0],  # DQN sử dụng hành động riêng lẻ
            reward=rewards_[:, idx],  # Reward của agent hiện tại
            next_state=actor_new_states[idx],
            done=terminal[:, idx]  # Trạng thái kết thúc của agent hiện tại
        )

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        # In kết quả trung bình
        if i % 10 == 0:
            print(f"Episode {i}, Score: {score}, Avg Score: {avg_score:.2f}")
        
        # Lưu checkpoint khi có kết quả tốt nhất
        if avg_score > best_score:
            best_score = avg_score
            print(f"New best score at episode {i}: {best_score:.2f}")
    # Vẽ đồ thị kết quả
    plt.figure(figsize=(10, 6))
    plt.plot(score_history)
    plt.title("Reward Trend Across Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()
