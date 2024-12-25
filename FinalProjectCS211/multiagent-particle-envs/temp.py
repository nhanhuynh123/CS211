import numpy as np
import os
import sys

reward_record_path = "/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/reward_records"
name = "ddpg_reward_record"
epochs = "50000"
files = [os.path.join(reward_record_path, f) for f in os.listdir(reward_record_path) if f.startswith(name+"_"+epochs) and not f.endswith("mean.npy")]

# print(files)

for i in range(len(files)):
    array1 = np.load(files[i])

    for j in range(i+1, len(files)):
        array2 = np.load(files[j])

        if np.array_equal(array1, array2):
            print(f"Files {files[i]} and {files[j]} are equal\n")