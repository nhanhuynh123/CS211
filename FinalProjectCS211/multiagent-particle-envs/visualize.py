import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
current_dir = os.path.dirname(os.path.abspath(__file__))  
reward_records_dir = os.path.join(current_dir, "reward_records")  

import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window=100):
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def plot(data):
    plt.figure(figsize=(10,5))
    plt.xlabel("Episode")
    plt.title("Score over time")
    plt.ylabel("Average Score")
    plt.grid()
    for name, score in data.items():
        average = moving_average(score, 10000)
        plt.plot(average, label=name)

    plt.legend()
    plt.show()
# 4, 11, 2, 3
#
file = ["maac_reward_record_4.npy", "maddpg_reward_record_mean.npy", "madqn_reward_record_2.npy", "ddpg_reward_record_3.npy"]
dir = []
for i in range(len(file)):
    dir.append(os.path.join(reward_records_dir, file[i]))
data = {
    "Actor Critic": np.load(dir[0]),
    "MADDPG": np.load(dir[1]),
    "DQN": np.load(dir[2]),
    "DDPG": np.load(dir[3])
}
plot(data)
