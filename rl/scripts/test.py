import gymnasium as gym
import torch
import torch.nn as nn

# Define the model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

# Initialize the environment
env = gym.make("LunarLander-v2", render_mode='human')

# Define dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# load model
dqn_model = DQN(state_dim, 32, action_dim)
dqn_model.load_state_dict(torch.load('LunarLander-v2_dqn.pt'))

# Testing the environment
for episode in range(10):
    state, _ = env.reset()
    episode_reward = 0
    terminated, truncated = False, False

    while not terminated and not truncated:
        states = torch.tensor(state, dtype=torch.float32)
        action = dqn_model(states).argmax().item()
        new_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
    print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")

# Closing the environment
env.close()
