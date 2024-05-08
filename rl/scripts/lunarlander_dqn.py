import gymnasium as gym
import torch, random
import torch.nn as nn
from collections import deque

# Define Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

# Define memory of experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)

# Initializing the game environment
env = gym.make('LunarLander-v2', render_mode='rgb_array')

# Constants
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
dqn_model = DQN(num_states, 32, num_actions)

# Hyperparameters
batch_size = 64
gamma = 0.95
learning_rate = 0.01
replay_buffer_size = 10000
num_episodes = 1000

# Initialize the replay buffer and optimizer
replay_buffer = ReplayBuffer(replay_buffer_size)
optimizer = torch.optim.Adam(dqn_model.parameters(), learning_rate)
loss_fn = nn.MSELoss()

# Training loop
for episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    terminated, truncated = False, False

    while not terminated and not truncated:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = dqn_model(state_tensor)
        action = torch.argmax(action_probs).item()

        # Take action and observe the result
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward

        # Sample a batch from the replay buffer and update the model 
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

            current_q_values = dqn_model(states).gather(1, actions)
            next_q_values = dqn_model(next_states).max(1)[0].unsqueeze(-1).detach()
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            # print("current_q_values shape:", current_q_values.shape,
            #       "next_q_values:", next_q_values.shape,
            #       "target_q_values shape:", target_q_values.shape)
            
            loss = loss_fn(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

    print(f"Episode {episode + 1}: Reward = {episode_reward}")
            
# Save the model weights
torch.save(dqn_model.state_dict(), 'LunarLander-v2_dqn.pt')

# Close the environment
env.close()

