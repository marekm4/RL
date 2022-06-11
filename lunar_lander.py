import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import torch.nn as nn

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)
video = VideoRecorder(env, "recording-ml.mp4")

HIDDEN_SIZE = 128

class Agent:
    def __init__(self):
        self.step = -1

    def policy(self, observation):
        self.step += 1
        velocity_v = -observation[3]
        velocity_h = observation[2]
        if abs(velocity_h) > .4 and self.step % 3 != 0:
            if velocity_h < 0:
                if self.step % 20 != 0:
                    return 0
                return 3
            elif velocity_h > 0:
                if self.step % 20 != 0:
                    return 0
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


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class AgentML:
    def __init__(self):
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.net = Net(obs_size, HIDDEN_SIZE, n_actions)
        self.net.load_state_dict(torch.load('model.pth'))

    def policy(self, observation):
        sm = nn.Softmax(dim=1)
        obs_v = torch.FloatTensor([observation])
        act_probs_v = sm(self.net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        if np.sum(observation[-2:]) == 2:
            return 0
        return np.argmax(act_probs)

episode = 0
total_rewards = [0] * 10
agent = AgentML()
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
