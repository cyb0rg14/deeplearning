import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
 
# Downlaod the data
dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=transforms.ToTensor())

# Divide the data into training & validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loader objects
dataloader = torch.utils.data.DataLoader
train_loader = dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = dataloader(val_dataset, batch_size=32, shuffle=True)
print(len(train_loader), len(val_loader))

# Create Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create device instance
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = CNN().to(device)
print(model)

# Initialize cost function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train and evaluate the model
from helper import train_and_eval

train_and_eval(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)
