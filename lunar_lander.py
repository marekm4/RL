import gym
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from dqn_agent import Agent

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)
video = VideoRecorder(env, "recording-ml.mp4")

HIDDEN_SIZE = 128

episode = 0
total_rewards = [0] * 10
agent = Agent(state_size=8, action_size=4, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
for _ in range(10000):
    env.render()
    video.capture_frame()
    action = agent.act(observation)  # User-defined policy function
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
