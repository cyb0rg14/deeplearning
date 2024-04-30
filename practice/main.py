import torch
from torchvision import models, transforms, datasets
from torchsummary import summary
from practice.helper import get_vgg16_model

model = get_vgg16_model(device='cpu')
summary(model, torch.zeros(1, 3, 224, 224))

