import gym
import time
import numpy as np

# Create an environment
env = gym.make('CartPole-v1', render_mode='human')

# Returns an intial state
state, _ = env.reset()

# Simulate the environment
episodes = 1000
time_steps = 100

for episodes in range(episodes):
    state, _ = env.reset()
    for time_steps in range(time_steps):
        env.render()
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        if done:
            break
