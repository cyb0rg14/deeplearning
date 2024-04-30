import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load & transform data
# dataset url -> https://www.kaggle.com/datasets/crownedhead06/big-cats-images-dataset
source_directory = '../../datasets/big-cats-images-dataset'
images_dataset = datasets.ImageFolder(source_directory, data_transforms['train'])
print(images_dataset)
images_dataset.class_to_idx

# Create data loaders
train_size = int(len(images_dataset) * 0.8)
val_size = len(images_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(images_dataset, [train_size, val_size]) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Create model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features # modify the last layer for our no of categories
model.fc = nn.Linear(num_ftrs, 9)

# Create cost fuction and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

from helper import train_and_eval

train_and_eval(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=num_epochs)
