import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np 
import json
import os
import sys
import time
import math
import io
from torchvision import models  
import torchvision.datasets as dsets    
import torch.nn.functional as F
import pandas as pd
from PIL import Image    
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = pd.read_csv(data_file, sep=" ", header=None, names=["image_path", "label"])
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]  
        image_path = os.path.join("./imagenet_dataset/data_val/", image_name)  #load imagenet_val
        label = int(self.data.iloc[idx, 1])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
data_file = "./imagenet_dataset/val.txt"   #imagenet_val label
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
val_dataset = CustomDataset(data_file, transform=val_transform)
batch_size = 1
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
model = models.resnet50(pretrained=True)
model.to(device)
model.eval()  
correct_images = []
correct_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted_labels = torch.argmax(outputs, dim=1)
        correct_mask = (predicted_labels == labels)
        correct_images.extend(images[correct_mask].cpu().numpy())
        correct_labels.extend(labels[correct_mask].cpu().numpy())
correct_images = torch.tensor(correct_images)  
correct_labels = torch.tensor(correct_labels)  
# Save a new dataset
torch.save({'images': correct_images, 'labels': correct_labels}, 'imagenet_dataset.pth')