import numpy as np
import matplotlib.pyplot as plt

_ = np.load("/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/MADDPG/_.npy")
__ = np.load("/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/MADDPG/__.npy")
___ = np.load("/Users/cicilian/Desktop/CS211/FinalProjectCS211/multiagent-particle-envs/MADDPG/___.npy")
# print(R.shape)
rgb_image = np.stack((_ / 255.0, __ /255.0, ___ / 255.0,), axis=-1)  # Kích thước sẽ là (height, width, 3)

plt.imshow(rgb_image)
plt.title("RGB Image")
plt.axis("off")
plt.show()