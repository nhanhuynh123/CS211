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
        average = moving_average(score, 1000)
        plt.plot(average, label=name)

    plt.legend()
    plt.show()
file = ["maac_reward_record.npy", "maddpg_reward_record.npy"]
dir = []
for i in range(len(file)):
    dir.append(os.path.join(reward_records_dir, file[i]))
data = {
    "Actor Critic": np.load(dir[0]),
    "MADDPG": np.load(dir[1])
}
plot(data)