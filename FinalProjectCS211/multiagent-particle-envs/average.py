import numpy as np
import os
import sys

reward_record_path = "/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/reward_records"
name = "maddpg_reward_record"
start = 1
files = []
while  True:
    dir = os.path.join(reward_record_path, f"{name}_{start}.npy")
    if os.path.exists(dir):
        files.append(dir)
    else:
        break
    start += 1

all_arrays = [np.load(file) for file in files]
for i in range(len(all_arrays)):
    print(all_arrays[i].shape)

# mean = np.mean(all_arrays, axis=0)

# np.save(os.path.join(reward_record_path, f"{name}_mean.npy"), mean)