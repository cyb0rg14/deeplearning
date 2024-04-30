from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms
from torchsummary import summary

import torch
import torch.nn
from torch.utils.data import DataLoader

device ="cuda:0" if torch.cuda.is_available() else "cpu"

# Transformations to apply
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device)),
])


# Download the dataset
train_ds = MNIST(root='../datasets/', train=True, transform=img_transform, download=True)
val_ds = MNIST(root='../datasets/', train=False, transform=img_transform, download=True)

# Define the dataloaders
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

# Define `Autoencoder` class
class ConvAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(ConvAutoEncoder, self).__init__()
        
        # Initialize encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        # Initialize decoder    
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Summary of the model
model = ConvAutoEncoder().to(device)
summary(model, torch.zeros(2, 1, 28, 28))

# Create cost fuction and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Define training function
def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

# Define validation function
@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss

num_epochs = 5
log = Report(num_epochs)


for epoch in range(num_epochs):
    N = len(train_loader)
    for ix, (data, _) in enumerate(train_loader):
        loss = train_batch(data, model, criterion, optimizer)
        log.record(pos=(epoch + (ix+1)/N), \
        trn_loss=loss, end='\r')
        N = len(val_loader)
    for ix, (data, _) in enumerate(val_loader):
        loss = validate_batch(data, model, criterion)
        log.record(pos=(epoch + (ix+1)/N), \
        val_loss=loss, end='\r')
    log.report_avgs(epoch+1)

# Visualize training and evaluation loss over each epoch
log.plot_epochs(log=True)

# Validate the model on test dataset
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1, 2, figsize=(3,3))
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()
    plt.show()


# Using t-SNE to cluster similar images together
latent_vectors = []
classes = []

for im, clss in val_loader:
    latent_vector = model.encoder(im).view(len(im), -1) 
    latent_vectors.append(latent_vector)
    classes.extend(clss)

latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()

# perform tsne
from sklearn.manifold import TSNE
tsne = TSNE(2)

clustered = tsne.fit_transform(latent_vectors)

# draw the cluster
fig = plt.figure(figsize=(12,10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
plt.colorbar(drawedges=True)
plt.show()