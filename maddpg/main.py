from buffer import MultiAgentReplayBuffer
from maddpg_torch import MADDPG
import numpy as np
import torch as T
from pettingzoo.mpe import simple_adversary_v3

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
        # print(obs)
    return state

if __name__ == '__main__':
    scenario = "simple_adversary"
    # T.autograd.set_detect_anomaly(True)
    env = simple_adversary_v3.env(continuous_actions=True)
    env.reset()
    n_agents = env.num_agents
    actor_dims = []

    for name in env.agents:
        actor_dims.append(env.observation_space(name).shape[0])
    critic_dims = sum(actor_dims)

    # n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, 5,
            fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario,
            chkpt_dir='tmp/maddpg/')
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
            5, n_agents, batch_size=1024)
    
    PRINT_INTERVAL = 500
    N_GAMES = 1000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()
    
    for i in range(N_GAMES):
        env.reset()
        score = 0
        dones = [False]*n_agents
        episode_step = 0
        obs = []
        for name in env.agents:
            obs.append(env.observe(name))

        while not any(dones):
            if evaluate:
                env.render()

            obs_ = []
            rewards = []
            dones = []
            noise_actions = []
            free_noise_actions = []
            next_actions = []

            for id, name in enumerate(env.agents):
                noise_action = maddpg_agents.choose_noise_action(env.observe(name), id)
                free_noise_action = maddpg_agents.choose_unnoise_action(env.observe(name), id)
                
                env.step(noise_action)
                
                noise_actions.append(noise_action)
                free_noise_actions.append(free_noise_action)
      
            for name in env.agents:
                obs_.append(env.observe(name))

            for key, value in env.rewards.items():
                rewards.append(value)

            temp_1 = []
            temp_2 = []
            for key, value in env.terminations.items():
                temp_1.append(value)

            for key, value in env.truncations.items():
                temp_2.append(value)
            temp_1 = np.array(temp_1)
            temp_2 = np.array(temp_2)

            dones = temp_1 | temp_2

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            # print("obs", obs)
            # print("noise act", noise_actions)
            # print("free noise act", free_noise_actions)
            # print("ep", i, rewards)
            # print(dones)
            # print("obs_", obs_)
            for id, name in enumerate(env.agents):
                # print(id, name)
                # print(env.observe(name))
                next_actions.append(maddpg_agents.choose_unnoise_action(env.observe(name), id))
                # print(name, env.observe(name))
            # print("next action", next_actions)
            if episode_step > MAX_STEPS:
                dones = [True]*n_agents

            memory.store_transition(state, noise_actions, free_noise_actions, rewards, state_, next_actions, dones)

            if total_steps % 100 == 0 and not evaluate:
                for id in range(env.num_agents):
                    if not memory.ready():
                        continue
                    bf_state, bf_noise_actions, bf_free_noise_actions, bf_rewards, bf_state_, bf_actions_, bf_done = memory.sample_buffer()
                    # print(bf_state.shape, bf_noise_actions.shape, bf_free_noise_actions.shape, bf_rewards.shape, bf_state_.shape, bf_actions_.shape, bf_done.shape )
                    maddpg_agents.learn(bf_state, bf_noise_actions, bf_free_noise_actions, bf_rewards, bf_state_, bf_actions_, bf_done, id)

            obs = obs_

            score += sum(rewards)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, ' average score {:.1f}'.format(avg_score))
 # print("noise_actions free_noise_actions rewards dones")
# print(noise_actions)
# print(free_noise_actions)

# print(env.rewards)