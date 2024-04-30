import torch
import torch.nn as nn
from torchvision import models, datasets
from tqdm import tqdm

# Train and evaluate for the model
def train_and_eval(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    for epoch in range(num_epochs):
        # Training phase
        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval() # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)

        print(f"[Epoch {epoch+1}/{num_epochs}] loss: {train_loss:.4f}, acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    print('\n\nTraining complete 🎉🎉'.upper())


# Return fine tuned vgg16 model
def get_vgg16_model(device, out_features=2):
    model = models.vgg16(pretrained=True).to(device)

    for params in model.parameters():
        params.requrires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, out_features=out_features)
    )

    return model


# Return fine tuned resnet18 model
def get_resnet18_model(device, out_features=2):
    model = models.resnet18(pretrained=True).to(device)

    for params in model.parameters():
        params.requrires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, out_features=out_features)
    )

    return model


# Return training and valdiation data loaders
def get_dataloader(transforms, images_dir, batch_size=64, train_size=0.8):
    dataset = datasets.ImageFolder(root=images_dir, transform=transforms)

    train_size = int(len(dataset) * train_size)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    dataloader = torch.utils.data.DataLoader
    train_loader = dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = dataloader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader