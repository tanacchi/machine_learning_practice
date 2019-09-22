import gym
import numpy as np

env = gym.make('FrozenLake-v0')
q_func = np.zeros((16, 4))

total_reward = 0.0
epsilon = 0.3
episodes = 10000

for i_episode in range(episodes):
    observation = env.reset()
    episode_reward = 0.0

    for t in range(100):
        current_state = observation
        
        if np.random.random() < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_func[current_state])

        observation, reward, done, info = env.step(action)
        update = epsilon * (reward + 0.99 * np.max(q_func[observation,:]) - q_func[current_state, action])
        q_func[current_state, action] += update

        if done:
            episode_reward += reward

    total_reward += episode_reward

print(total_reward / epsilon)
print(q_func)

observation = env.reset()
episode_reward = 0.0

print("=====================")

for t in range(100):
    env.render()
    current_state = observation
    action = np.argmax(q_func[current_state])
    observation, reward, done, info = env.step(action)

    if done:
        episode_reward += reward
