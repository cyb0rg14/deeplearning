# Import all necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models, datasets
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np, pandas as pd, os, glob, cv2
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cluster
device = "cuda:0" if torch.cuda.is_available() else "cpu"

root_dir = "./P1_Facial_Keypoints/data/training/"
all_img_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
# all_img_paths[:5]

# paste (git clone https://github.com/udacity/P1_Facial_Keypoints.git) in your terminal to install this dataset
data = pd.read_csv("../P1_Facial_Keypoints/data/training_frames_keypoints.csv")
# print(data.head())

# Define FacesData class that provides input and output data points for the data loader
class FacesData(Dataset):
    def __init__(self, df) -> None:
        super(FacesData, self).__init__()
        self.df = df
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int: 
        return len(self.df)

    def __getitem__(self, index):
        img_path = root_dir + self.df.iloc[index, 0] # returns the image path
        img = cv2.imread(img_path)/255.0 # scale the image

        # Set the keypoints as proportion of the original image, so when we resize, it doesn't change the actual location of keypoints
        kp = deepcopy(self.df.iloc[index, 1:].tolist())
        kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2])/img.shape[0]).tolist()

        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2
    
    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize(img).float()
        return img
    
    def load_img(self, idx):
        img_path = root_dir + self.df.iloc[idx, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        img = cv2.resize(img, (224, 224))
        return img


# Let's create a training & test data split and establish data loaders
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=101)

train_dataset = FacesData(train.reset_index(drop=True))
test_dataset = FacesData(test.reset_index(drop=True))

train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Load pretrained VGG16 model
def get_model():
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # Overwrite and unfreezer the parameters of the last two layers
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten()
    )

    model.classifier = nn.Sequential(
        nn.Linear(4608, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()
    )

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model.to(device), criterion, optimizer

# Get model, criterion & optimizer
model, criterion, optimizer = get_model()

# Define a training function
def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss

# Define a validation function
def validate_batch(img, kps, model, criterion):
    model.eval()
    with torch.no_grad():
        _kps = model(img.to(device))
        loss = criterion(_kps, kps.to(device))
        return _kps, loss

# Training and testing time
train_loss, test_loss = [], []
n_epochs = 50
for epoch in range(n_epochs):
    print(f" epoch {epoch+ 1} : 50")
    epoch_train_loss, epoch_test_loss = 0, 0
    for ix, (img,kps) in enumerate(train_dataloader):
        loss = train_batch(img, kps, model, optimizer, criterion)
        epoch_train_loss += loss.item()
        epoch_train_loss /= (ix+1)
    for ix,(img,kps) in enumerate(test_dataloader):
        ps, loss = validate_batch(img, kps, model, criterion)
        epoch_test_loss += loss.item()
        epoch_test_loss /= (ix+1)
    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)

# from helper import train_and_eval
# train_and_eval(model, criterion, optimizer, train_dataloader, test_dataloader, device, num_epochs=50)

epochs = np.arange(50)+1
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, test_loss, 'r', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

ix = 0
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.title('Original image')
im = test_dataset.load_img(ix)
plt.imshow(im)
plt.grid(False)
plt.subplot(222)
plt.title('Image with facial keypoints')
x, _ = test_dataset[ix]
plt.imshow(im)
kp = model(x[None]).flatten().detach().cpu()
plt.scatter(kp[:68]*224, kp[68:]*224, c='r')
plt.grid(False)
plt.show()
