import torch
import torch.nn as nn
import torch.optim as optim 
from boilerplate import *

# source directory
source_directory = "../datasets/big-cats-images-dataset"

# load the images
dataset = images_dataset(source_directory)

# train and validation dataloaders
train_loader, val_loader = dataloader(dataset, 0.8, 32)

# first batch
first_batch = next(iter(train_loader))

# displaying first batch
# display_images_from_batch(first_batch)

# defining the model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Initializing the model, optimizer and loss function
num_classes = len(dataset.classes)
model = SimpleCNN(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training the model
num_epochs = 10
print("Training has started ...")
train_and_eval(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs)
