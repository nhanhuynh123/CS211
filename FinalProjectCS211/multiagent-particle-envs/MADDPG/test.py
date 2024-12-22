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

# Đường dẫn tới checkpoint đã lưu
CHECKPOINT_DIR = "tmp/maddpg/"

# Hàm trực quan hóa
def visualize_model():
    # Tạo môi trường
    scenario = "simple_adversary"
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=True, render_mode="human")
    obs = env.reset()

    # Số lượng agent và các tham số của mô hình
    n_agents = env.num_agents
    actor_dims = [env.observation_space(agent_name).shape[0] for agent_name in env.agents]
    critic_dims = sum(actor_dims)
    n_actions = 5

    # Khởi tạo mô hình
    maddpg_agents = MADDPG(
        actor_dims, critic_dims, n_agents, n_actions,
        fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario, chkpt_dir=CHECKPOINT_DIR
    )

    # Load checkpoint
    print("Loading model from checkpoint...")
    maddpg_agents.load_checkpoint()
    print("Checkpoint loaded successfully.")

    # Chạy mô hình
    print("Starting visualization...")
    num_episodes = 5  # Số lượng tập muốn trực quan hóa
    scores = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        done = [False] * n_agents
        score = 0
        step = 0

        while not any(done):
            env.render()
            time.sleep(0.7)  # Giảm tốc độ hành động để dễ quan sát

            # Lựa chọn hành động từ mô hình
            actions = maddpg_agents.choose_action(obs)
            obs_, rewards, terminations, truncations, _ = env.step(actions)

            score += sum(rewards.values())
            step += 1

            # Kiểm tra điều kiện kết thúc
            if any(terminations.values()) or any(truncations.values()):
                done = [True] * n_agents

            obs = obs_

        scores.append(score)
        print(f"Episode {episode + 1}/{num_episodes} | Score: {score:.2f} | Steps: {step}")

    env.close()
@dataclass
class Args:
    epochs: int  = 40000
    # Number games
    PRINT_INTERVAL: int = 500
    # Print frequency
    MAX_STEPS: int =25
    # Max episode steps
    demo = False

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "tmp/maddpg")
    args = tyro.cli(Args)
    # scenario = 'simple'
    scenario = 'simple_adversary'
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=40, continuous_actions=True, render_mode ="human")
    obs = env.reset()

    n_agents = env.num_agents
    actor_dims = []
    for agent_name in env.agents:
        actor_dims.append(env.observation_spaces[agent_name]._shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 5
    dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "tmp/maddpg")
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
    maddpg_agents.load_checkpoint()
    time.sleep(1)
    for episode in range(args.epochs):
        obs, _ = env.reset(seed=episode)
        done = [False] * n_agents
        score = 0
        step = 0

        while not any(done):
            env.render()
            time.sleep(0.7)  # Giảm tốc độ hành động để dễ quan sát

            # Lựa chọn hành động từ mô hình
            actions = maddpg_agents.choose_action(obs)
            obs_, rewards, terminations, truncations, _ = env.step(actions)

            score += sum(rewards.values())
            step += 1

            # Kiểm tra điều kiện kết thúc
            if any(terminations.values()) or any(truncations.values()):
                done = [True] * n_agents

            obs = obs_

        score.append(score)
        print(f"Episode {episode + 1}/{episode} | Score: {score:.2f} | Steps: {step}")
    env.close()