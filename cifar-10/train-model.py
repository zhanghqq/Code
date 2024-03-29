import numpy as np 
import json
import os
import sys
import time
import math
import io 
import torch.optim as optim 
from torchvision import models  
import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
from  torchattacks.attack import Attack  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
train_transform = transforms.Compose([
#     tt.RandomHorizontalFlip(p=0.5),
#     tt.RandomVerticalFlip(p=0.5),
   transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.ToTensor(),
   transforms.Normalize(
                [0.485, 0.456, 0.406],     #
                [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
transforms.Normalize(
                [0.485, 0.456, 0.406],     #
                [0.229, 0.224, 0.225])
])

class BasicBlock(nn.Module):
   
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])
def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform,
                                         download=False)
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform,
                                         download=False)
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=64)
test = ResNet50() 
test = test.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)
total_train_step = 0
total_test_step = 0
epoch = 60
for i in range(epoch):
    print("-----iter{}------".format(i+1))
    test.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = test(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("steps{}, Loss:{}".format(total_train_step, loss.item()))
    test.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = test(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
 
    print("Loss: {}".format(total_test_loss))
    print(":Accuracy {}".format(total_accuracy/test_data_size))
    total_test_step += 1
    if (i+1)%10==0:
        torch.save(test, "resnet50_{}.pth".format(i))
        print("Model Saved")
