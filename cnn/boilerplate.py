import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import torch.nn.functional as F 

import numpy as np
# import matplotlib.pyplot as plt

transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def images_dataset(source_directory, transforms=transforms):
    """
    Create a dataset of images from the source directory.

    Parameters:
        source_directory (str): Path to the dataset directory.
        transforms (callable, optional): A composition of image transformations (default: predefined transforms).

    Returns:
        torchvision.datasets.ImageFolder: Dataset of images from the source directory.

    Notes:
        - This function creates a dataset of images using the torchvision.datasets.ImageFolder class.
        - The dataset can be augmented with image transformations provided via the 'transforms' parameter.
        - It allows using all the methods associated with the ImageFolder dataset.

    Example:
        # Create a dataset of images with transformations
        dataset = images_dataset(source_directory='./data', transforms=transforms)
    """
    return datasets.ImageFolder(source_directory, transform=transforms)

def dataloader(dataset, training_fraction, batch_size):
    """
    Create train and validation dataloaders from the given dataset.

    Parameters:
        dataset: Dataset of images.
        training_fraction (float): Fraction of the dataset to use for training.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the training dataset.
        torch.utils.data.DataLoader: Dataloader for the validation dataset.

    Notes:
        - This function splits the provided dataset into training and validation sets based on the specified fraction.
        - The training and validation dataloaders are created with the given batch size and are shuffled.

    Example:
        # Create dataloaders for the dataset
        train_loader, val_loader = dataloader(dataset, training_fraction=0.8, batch_size=32)
    """
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

def display_images_from_batch(batch):
    """
    Display images and their corresponding labels from a given batch in a grid-like structure.

    Parameters:
        batch (tuple): A tuple containing images and their corresponding labels.

    Returns:
        None

    Notes:
        This function assumes that the images are normalized with mean [0.485, 0.456, 0.406]
        and standard deviation [0.229, 0.224, 0.225], as commonly used in PyTorch's torchvision
        library.

        The function will automatically adjust the grid size based on the number of images in the batch,
        displaying images in 5 columns and a dynamically determined number of rows to accommodate all images.

    Example:
        # Assuming you have a batch obtained from your dataloader
        display_images_from_batch(batch)
    """
    images, labels = batch
    images, labels = images.numpy(), labels.numpy() # convert tensors into numpy arrays
    batch_size = len(images) # determine the grid size based on the batch size
    rows = int(np.ceil(batch_size / 5))  # Display images in 5 columns
    fig, axes = plt.subplots(rows, 5, figsize=(15, rows*3)) # plot the images
    fig.subplots_adjust(hspace=0.5)
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            image = images[i]
            # Denormalize the image
            image = np.transpose(image, (1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.set_title(f'Class: {labels[i]}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.show()

def train_and_eval(model, criterion, optimizer, train_loader, val_loader, device="cpu", num_epochs=10):
    """
    Train and evaluate a given model using the provided data loaders.

    Parameters:
        model (torch.nn.Module): The neural network model to train and evaluate.
        criterion: The loss function used to compute the loss.
        optimizer: The optimizer used to update the model parameters.
        train_loader (torch.utils.data.DataLoader): Data loader for the training dataset.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation dataset.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        num_epochs (int, optional): Number of epochs to train the model (default is 10).

    Returns:
        None

    Notes:
        - This function trains the model for the specified number of epochs using the training data loader.
        - After each epoch, it evaluates the model's performance on the validation data loader.
        - The model is set to training mode during the training phase and evaluation mode during the validation phase.
        - It prints the training and validation loss and accuracy for each epoch.

    Example:
        # Define model, criterion, optimizer, and data loaders
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train and evaluate the model
        train_and_eval(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)
    """
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

