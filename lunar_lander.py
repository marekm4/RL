import gym
import numpy as np

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)


def policy(observation):
    velocity_v = -observation[3]
    velocity_h = observation[2]
    if velocity_v > .6:
        if abs(velocity_h) < .3:
            return 2
        else:
            if velocity_h < 0:
                return 3
            elif velocity_h > 0:
                return 1
    return 0


episode = 0
total_rewards = [0] * 10
for _ in range(10000):
    env.render()
    action = policy(observation)  # User-defined policy function
    observation, reward, done, info = env.step(action)
    total_rewards[episode] += reward

    if done:
        observation, info = env.reset(return_info=True)
        episode += 1
    if episode == 10:
        break
env.close()

print(total_rewards)
print(np.average(total_rewards))
