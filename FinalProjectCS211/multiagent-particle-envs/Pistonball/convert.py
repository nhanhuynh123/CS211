from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=False, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125, render_mode="")

import supersuit as ss

env = ss.color_reduction_v0(env, mode="full")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
import numpy as np
import matplotlib.pyplot as plt

obs, _ = env.reset()   
print(obs["piston_0"].shape)

name = "piston_18"
for i in range(3):
    action = {name: env.action_spaces[name].sample() for name in env.possible_agents}
    obs, reward, ter, trunc, info = env.step(action)   
    r = obs[name][:,:,0]
    g = obs[name][:,:,1]
    b = obs[name][:,:,2]
    f = obs[name][:,:,3]
    # r = obs[name]
# print(np.array_equal(b, r))

image = np.stack([r, g, b, f], axis=-1)
# image = np.stack([r], axis=-1)

plt.imshow(image)
plt.show()