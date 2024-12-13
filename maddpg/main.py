from buffer import MultiAgentReplayBuffer
from maddpg_torch import MAAC  # Assuming you have updated this to MAAC
import numpy as np
import torch as T
from pettingzoo.mpe import simple_adversary_v3

def obs_list_to_state_vector(observation):
    state = np.array([])  # Flatten the observations into a state vector
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    scenario = "simple_adversary"
    env = simple_adversary_v3.env(continuous_actions=True)
    env.reset()

    n_agents = env.num_agents
    actor_dims = [env.observation_space(name).shape[0] for name in env.agents]
    critic_dims = sum(actor_dims)

    # Initialize MAAC agents
    maac_agents = MAAC(actor_dims, critic_dims, n_agents, 5, fc1=128, fc2=128, alpha=0.001, beta=0.01,
                      scenario=scenario, chkpt_dir='tmp/maac/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 5, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maac_agents.load_checkpoint()

    for i in range(N_GAMES):
        env.reset()
        score = 0
        dones = [False] * n_agents
        episode_step = 0
        obs = [env.observe(name) for name in env.agents]

        while not any(dones):
            if evaluate:
                env.render()

            obs_ = []
            rewards = []
            dones = []
            noise_actions = []
            next_actions = []

            for id, name in enumerate(env.agents):
                # MAAC chooses deterministic actions (no noise) during evaluation
                action = maac_agents.choose_action(env.observe(name))  # No noise added here

                env.step(action)  # Execute the action
                noise_actions.append(action)  # Track the actions for training

            for name in env.agents:
                obs_.append(env.observe(name))

            rewards = list(env.rewards.values())
            dones = np.array(list(env.terminations.values())) | np.array(list(env.truncations.values()))

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # Collect actions for the next step (deterministic)
            for id, name in enumerate(env.agents):
                next_actions.append(maac_agents.choose_action(env.observe(name), id))

            if episode_step > MAX_STEPS:
                dones = [True] * n_agents

            # Store the experience in memory
            memory.store_transition(state, noise_actions, noise_actions, rewards, state_, next_actions, dones)

            if total_steps % 100 == 0 and not evaluate:
                for id in range(env.num_agents):
                    if not memory.ready():
                        continue
                    # Sample a batch from the memory buffer
                    bf_state, bf_noise_actions, bf_free_noise_actions, bf_rewards, bf_state_, bf_actions_, bf_done = memory.sample_buffer()
                    maac_agents.learn(bf_state, bf_noise_actions, bf_free_noise_actions, bf_rewards, bf_state_, bf_actions_, bf_done, id)

            obs = obs_

            score += sum(rewards)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maac_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, ' average score {:.1f}'.format(avg_score))
