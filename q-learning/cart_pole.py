import gym
from gym import wrappers
import numpy as np
import time

def discretize(begin, end, num):
    return np.linspace(begin, end, num + 1)[1:-1]


def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos,   bins=discretize(-2.4, 2.4, num_digitized)),
        np.digitize(cart_v,     bins=discretize(-3.0, 3.0, num_digitized)),
        np.digitize(pole_angle, bins=discretize(-0.5, 0.5, num_digitized)),
        np.digitize(pole_v,     bins=discretize(-2.0, 2.0, num_digitized))
    ]
    return sum([x * (num_digitized**i) for i, x in enumerate(digitized)])


def get_action(next_state, episode):
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        return np.argmax(q_table[next_state])
    else:
        return np.random.choice([0, 1])


def update_q_table(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_max_q = max(q_table[next_state][0], q_table[next_state][1])
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * next_max_q)
    return q_table


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    max_steps           = 200
    iterations          = 100
    episodes            = 2000
    goal_avarage_reward = 195
    num_digitized       = 8
    q_table = np.random.uniform(low=-1, high=1, size=(num_digitized**4, env.action_space.n))
    total_reward_vec      = np.zeros(iterations)
    final_x               = np.zeros((episodes, 1))
    is_learned, is_render = False, False

    for episode in range(episodes):
        observation = env.reset()
        state = digitize_state(observation)
        action = np.argmax(q_table[state])
        episode_reward = 0

        for t in range(max_steps):
            if episode % 100 == 0:
                env.render()
                time.sleep(0.05)
                print("[{}]\t=>{}:\t{}".format(episode, t, observation))

            observation, reward, done, info = env.step(action)

            if done:
                if t < 195:
                    reward = -200
                else:
                    reward = 5 - abs(observation[0])*2
            else:
                reward = 1

            episode_reward += reward

            next_state = digitize_state(observation)
            q_table = update_q_table(q_table, state, action, reward, next_state)
            action = get_action(next_state, episode)
            state = next_state

            if done:
                print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))
                if is_learned:
                    final_x[episode, 0] = observation[0]
                break

            if total_reward_vec.mean() >= goal_avarage_reward:
                is_learned = True
                if is_render == False:
                    is_render = True
        env.close()
