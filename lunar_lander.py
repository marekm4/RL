import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)
video = VideoRecorder(env, "recording.mp4")

class Agent:
    def __init__(self):
        self.step = -1

    def policy(self, observation):
        self.step += 1
        velocity_v = -observation[3]
        velocity_h = observation[2]
        if abs(velocity_h) > .4 and self.step % 3 != 0:
            if velocity_h < 0:
                return 3
            elif velocity_h > 0:
                return 1
        if velocity_v > .6:
            if abs(velocity_h) < .3 or self.step % 4 != 0:
                return 2
            else:
                if velocity_h < 0:
                    return 3
                elif velocity_h > 0:
                    return 1
        return 0


episode = 0
total_rewards = [0] * 10
agent = Agent()
for _ in range(10000):
    env.render()
    video.capture_frame()
    action = agent.policy(observation)  # User-defined policy function
    observation, reward, done, info = env.step(action)
    total_rewards[episode] += reward

    if done:
        observation, info = env.reset(return_info=True)
        episode += 1
    if episode == 10:
        break
video.close()
env.close()

print(total_rewards)
print(np.average(total_rewards))
