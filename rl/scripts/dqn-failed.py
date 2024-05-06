from os import truncate
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model
class DQN(nn.Module):
    def __init__(self, in_states: int, h1_nodes: int, out_actions: int) -> None:
        super().__init__() 
        # Define network layers 
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define memory for experience replay
class ReplayMemory:
    def __init__(self, maxlen: int) -> None:
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, k=sample_size)

    def __len__(self):
        return len(self.memory)

# FrozenLake Deep Q-Learning
class FrozenLakeDQL:
    # Hyperparameters (adjustable)
    learning_rate = 0.01
    discount_rate = 0.95
    network_sync_rate = 10
    replay_memory_size = 1000
    sample_size = 32

    # Neural networks
    loss_fn = nn.MSELoss()
    optimizer = None

    actions = ['l', 'd', 'r', 'u']

    # Train the frozenlake environment
    def train(self, episodes, render=False, is_slippery=False):
        # Create frozenlake instance
        env = gym.make("FrozenLake-v1", map_name='4x4', is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n      
        num_actions = env.action_space.n
        
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target networks
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Copy weights/biases from policy to target
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        print("Random Policy Before Training ...")
        self.print_dqn(policy_dqn) 
    
        # Optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate) 
        
        # Keep track of rewards per episode
        rewards_per_episode = np.zeros(episodes)

        # Keep track of epsilon decay
        epsilon_history = [] 

        # Track step count (used for syncing policy -> target network)
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0] # Initialize to state 0
            terminated = False # Reached the terminal state
            truncated = False # Reached the max steps (maybe 200)

            # Agent navigate through the map until it get terminated/trucated
            while (not terminated or not truncated):
                # Select action based on epsilon greedy policy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                else:
                    # select best action
                    with torch.no_grad():
                        action =  policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                
                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))
                
                # Move to the next state 
                state = new_state

                # Increase step counter
                step_count += 1
                
            # Keep track of rewards per episodes
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if
            if len(memory) > self.sample_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.sample_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                
                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after certain steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
        
        # Close the environment            
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozenlake_dqn.pt")

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-Axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
    
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
    
        # Save plots
        plt.savefig('frozen_lake_dql.png')


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

            
            
    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("frozenlake_dqn.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()
        

    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.actions[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states  


    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor


if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    frozen_lake.train(1000) 
    frozen_lake.test(10)
