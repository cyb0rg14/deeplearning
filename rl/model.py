import gym
import random

# Create the environment
env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

# Checking how many states and actions there are
print(f"States: {states}, Actions: {actions}")

# Running the environment
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # Render the environment
        env.render()
        # Choose a random action
        action = random.choice([0,1])
        # Take the action and get the new state, reward, done flag, and info
        state, reward, done, info, _ = env.step(action)
        score += reward

    print(f'Episode: {episode}, Score: {score}')

# Close the environment after the loop
env.close()

# Create a deep learning model with keras
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)
model.summary()

# Build agent with keras-rl
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Testing the agent
scores = dqn.test(env, nb_episodes=10, visualize=False)
