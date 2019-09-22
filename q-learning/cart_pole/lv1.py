import gym
from gym import wrappers
import time

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    max_steps = 200
    episodes = 10

    for episode in range(episodes):
        observation = env.reset()
        total_reward = 0

        for t in range(max_steps):
            env.render()
            time.sleep(0.1)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                if t + 1 == max_steps:
                    reward = 100
                else:
                    reward = -100
            else:
                reward = 1
            total_reward += reward

            if done:
                break
        print("Fin: {}".format(total_reward))

    env.close()
