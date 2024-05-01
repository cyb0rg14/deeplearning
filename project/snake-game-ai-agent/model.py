import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        

class QTrainer:
    def __init__(self, lr, gamma, model):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # 1: predict Q values with current state
        states = states.to(DEVICE)
        pred = self.model(states)

        # 2: Q_new = r + y * max(predicted Q values of next state)
        target = pred.clone().to(DEVICE)
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))

            target[idx][torch.argmax(actions).item()] = Q_new

        # update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
