import numpy as np
import os
import sys

reward_record_path = "/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/reward_records"
name = "madqn_reward_record"
epochs = "50000"
files = [os.path.join(reward_record_path, f) for f in os.listdir(reward_record_path) if f.startswith(name+"_"+epochs) and not f.endswith("mean.npy")]
print(files)
# print(files[3:4])
all_arrays = [np.load(file) for file in files]
for  array in all_arrays:
    print(array.shape)
mean = np.mean(all_arrays, axis=0)

np.save(os.path.join(reward_record_path, f"{name}_{epochs}_mean.npy"), mean)