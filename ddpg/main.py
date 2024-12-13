from ddpg_multiagent import DDPGMultiAgent
import numpy as np
import torch as T
from pettingzoo.mpe import simple_adversary_v3

def obs_to_state(observation):
    return np.array(observation, dtype=np.float32)

if __name__ == '__main__':
    scenario = "simple_adversary"
    env = simple_adversary_v3.env(continuous_actions=True)
    env.reset()
    n_agents = env.num_agents
    actor_dims = []
    for name in env.agents:
        actor_dims.append(env.observation_space(name).shape[0])
    critic_dims = actor_dims  # Trong DDPG, Critic chỉ nhận state của chính mình

    n_actions = env.action_space(env.agents[0]).shape[0]

    ddpg_agents = DDPGMultiAgent(actor_dims, critic_dims, n_agents, n_actions,
                                 scenario=scenario,
                                 alpha=0.001, beta=0.001,
                                 fc1=400, fc2=300,
                                 gamma=0.99, tau=0.005,
                                 chkpt_dir='tmp/ddpg_multiagent/')
    
    PRINT_INTERVAL = 500
    N_GAMES = 1000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = -np.inf

    if evaluate:
        ddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        env.reset()
        score = 0
        dones = [False] * n_agents
        episode_step = 0
        obs = []
        for name in env.agents:
            obs.append(env.observe(name))

        while not any(dones):
            if evaluate:
                env.render()

            obs_ = []
            rewards = []
            dones_temp = []
            noise_actions = []
            unnoise_actions = []

            for id, name in enumerate(env.agents):
                noise_action = ddpg_agents.choose_noise_action(obs[id], id)
                unnoise_action = ddpg_agents.choose_unnoise_action(obs[id], id)
                noise_actions.append(noise_action)
                unnoise_actions.append(unnoise_action)

            actions = noise_actions  # Sử dụng hành động có nhiễu để khám phá

            # Thực hiện các hành động trong môi trường
            env.step(actions)

            for id, name in enumerate(env.agents):
                obs_.append(env.observe(name))
                rewards.append(env.rewards[name])
                dones_temp.append(env.terminations[name] or env.truncations[name])

            for id in range(n_agents):
                # Lưu trữ chuyển tiếp vào bộ nhớ của từng agent
                ddpg_agents.learn(obs[id], actions[id], rewards[id], obs_[id], dones_temp[id], id)

            obs = obs_

            score += sum(rewards)
            total_steps += 1
            episode_step += 1

            if episode_step > MAX_STEPS:
                dones = [True] * n_agents

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                ddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('Episode:', i, 'Average Score {:.1f}'.format(avg_score))
