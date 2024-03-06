import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import copy
import sys
import time
import math
import io
import pandas as pd
from torchvision import models  
import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
from  torchattacks.attack import Attack  
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np 
import json
from utils1 import *
from compression import *
from decompression import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#####AdvDrop Attack
class InfoDrop(Attack):      
    r"""    
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`. 
        
    """
    def __init__(self, model, height =224, width =224,  steps=50, batch_size = 32, block_size = 8, q_size = 100, targeted = False):
        super(InfoDrop, self).__init__("InfoDrop", model)     
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Value for quantization range
        self.factor_range = [5, q_size]
        # Differential quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)   
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size)                         
        q_ini_table = np.empty((batch_size,int(block_n),block_size,block_size), dtype = np.float32)   
        q_ini_table.fill(q_size)
        self.q_tables = {"y": torch.from_numpy(q_ini_table),     
                        "cb": torch.from_numpy(q_ini_table),
                        "cr": torch.from_numpy(q_ini_table)}        
    def forward(self, images, labels):   
        r"""
        Overridden.
        """
        q_table = None         
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_loss =  nn.CrossEntropyLoss()    
        optimizer = torch.optim.Adam([self.q_tables["y"],  self.q_tables["cb"], self.q_tables["cr"]], lr= 0.01)#优化
        images = images.permute(0, 2, 3, 1)  
        components = {'y': images[:,:,:,0], 'cb': images[:,:,:,1], 'cr': images[:,:,:,2]}
        for i in range(self.steps):
            self.q_tables["y"].requires_grad = True
            self.q_tables["cb"].requires_grad = True   
            self.q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = block_splitting(components[k])  
                comp = dct_8x8(comp)        
                comp = quantize(comp, self.q_tables[k], self.alpha)  
                comp = dequantize(comp, self.q_tables[k]) 
                comp = idct_8x8(comp)      
                merge_comp = block_merging(comp, self.height, self.width)   
                upresults[k] = merge_comp
            rgb_images = torch.cat([upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            rgb_images = rgb_images.permute(0, 3, 1, 2) 
            tool=transforms.Normalize(
                [0.485, 0.456, 0.406],    
                [0.229, 0.224, 0.225])
            nomal_images=tool(rgb_images/255.0)
            outputs = self.model(nomal_images)  
            _, pre = torch.max(outputs.data, 1)
            if self.targeted:
                suc_rate = ((pre == labels).sum()/self.batch_size).cpu().detach().numpy()   
            else:
                suc_rate = ((pre != labels).sum()/self.batch_size).cpu().detach().numpy()
            adv_cost = adv_loss(outputs, labels) 
            if not self.targeted:
                adv_cost = -1* adv_cost
            total_cost = adv_cost
            optimizer.zero_grad()
            total_cost.backward()
            self.alpha += self.alpha_interval   
            for k in self.q_tables.keys():
                with torch.no_grad(): 
                    self.q_tables[k] = self.q_tables[k].detach() -  torch.sign(self.q_tables[k].grad)
                    self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()
            if i%10 == 0:     
                print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate )
            if suc_rate >= 1 or i==49:# steps=50
                print('End at step {} with suc. rate {}'.format(i, suc_rate))
                q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
                return q_images, labels, i        
        q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
        return q_images, labels, q_table
###Calculate the spectral energy ratio
def energy(magnitude_spectrum,K=84):    #K=84 input_size=207
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    radius_max = min(center_row, center_col)
    radius_range = np.arange((K/2)-1, radius_max) 
    energy_ratios = []
    prev_energy_diff = np.sum(magnitude_spectrum)  
    for i in range(1, len(radius_range) - 1):
        mask_r_minus_1 = np.zeros_like(magnitude_spectrum, dtype=np.uint8)
        mask_r = np.zeros_like(magnitude_spectrum, dtype=np.uint8)
        mask_r_plus_1 = np.zeros_like(magnitude_spectrum, dtype=np.uint8)
        y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
        mask_r_minus_1_area = np.logical_and(np.abs(x) <= radius_range[i - 1], np.abs(y) <= radius_range[i - 1])
        mask_r_area = np.logical_and(np.abs(x) <= radius_range[i], np.abs(y) <= radius_range[i])
        mask_r_plus_1_area = np.logical_and(np.abs(x) <= radius_range[i + 1], np.abs(y) <= radius_range[i + 1])
        mask_r_minus_1[mask_r_minus_1_area] = 1
        mask_r[mask_r_area] = 1
        mask_r_plus_1[mask_r_plus_1_area] = 1
        energy_r_minus_1 = np.sum(magnitude_spectrum[mask_r_minus_1 == 1])
        energy_r = np.sum(magnitude_spectrum[mask_r == 1])
        energy_r_plus_1 = np.sum(magnitude_spectrum[mask_r_plus_1 == 1])
        energy_diff_r =  energy_r_minus_1-energy_r
        energy_diff_r_plus_1 = energy_r-energy_r_plus_1
        energy_ratio = energy_diff_r_plus_1 / energy_diff_r if energy_diff_r != 0 else 0
        energy_ratios.append(energy_ratio)
        prev_energy_diff = energy_diff_r_plus_1
    return radius_range,energy_ratios
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
q_size =100    
cur_cnt = 0
suc_cnt = 0  
global total
total=0
list1=[]
list2=[]
list=[]
total_train_step=0
test= models.resnet50(pretrained=True)   
test = test.to(device)
test.eval()
##Generate a training set for training the REAR detector
class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = transforms.ToPILImage()(image) 
            image = self.transform(image)
        return image, label
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),      
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.RandomResizedCrop(224),  
    transforms.ToTensor(),
])        
correctdataset = torch.load('./imagenet_dataset.pth')
correct_images = correctdataset['images']
correct_labels = correctdataset['labels']
# create the DataLoader
batch_size = 32
correct_dataset = torch.utils.data.TensorDataset(correct_images, correct_labels)
augmented_dataset = AugmentedDataset(correct_dataset, transform=augmentation_transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
# train_data_size = len(augmented_dataset)
# print("traindata-length{}".format(train_data_size)) #27708
all_energy_ratios = []
all_labels = []              
test.eval()
sum_image=[0,1,2]
for i in sum_image:
    total_train_step=0
    for data in augmented_loader:
        if total_train_step==865:  #27708/32=865
            break
        total_train_step+=1
        imgs,targets = data  
        print("Iter: ",total_train_step )
        imgs = imgs.detach().clone()
        targets = targets.detach().clone()
        imgs = imgs.to(device)
        targets = targets.to(device)
        imgs=imgs*255.0 
        # For target attack: set random target. 
        # Comment if you set untargeted attack.
        # labels = torch.from_numpy(np.random.randint(0, 1000, size = batch_size))
        attack = InfoDrop(test, batch_size=32, q_size =100, steps=50, targeted =False)
        at_images, labels, suc_step = attack(imgs, targets)  
        at_images = at_images.detach().clone()
        at_images =at_images.to(device)  
        clean_images = imgs     
        adversarial_images = at_images  
        clean_images_numpy = clean_images.detach().cpu().numpy()  
        adversarial_images_numpy = adversarial_images.detach().cpu().numpy() 
        clean_images_numpy = clean_images_numpy[0]
        clean_images_numpy= np.moveaxis(clean_images_numpy, 0, 2)
        adversarial_images_numpy  =  adversarial_images_numpy[0]
        adversarial_images_numpy  = np.moveaxis(adversarial_images_numpy , 0, 2)
        (B, G, R) = cv2.split(clean_images_numpy)   
        fR = np.fft.fft2(R)
        fG = np.fft.fft2(G)
        fB = np.fft.fft2(B)
        fshiftR = np.fft.fftshift(fR)
        fshiftG = np.fft.fftshift(fG)
        fshiftB = np.fft.fftshift(fB)
        fimgR = np.log(np.abs(fshiftR))
        fimgG = np.log(np.abs(fshiftG))
        fimgB = np.log(np.abs(fshiftB))
        radius_range_clean, energy_ratios_R = energy(fimgR,K=84)
        radius_range_clean, energy_ratios_G = energy(fimgG,K=84)
        radius_range_clean, energy_ratios_B = energy(fimgB,K=84)
        max_length = max(len(energy_ratios_R), len(energy_ratios_G), len(energy_ratios_B))
        energy_ratios_R = np.pad(energy_ratios_R, (0, max_length - len(energy_ratios_R)))
        energy_ratios_G = np.pad(energy_ratios_G, (0, max_length - len(energy_ratios_G)))
        energy_ratios_B = np.pad(energy_ratios_B, (0, max_length - len(energy_ratios_B)))
        energy_ratios_combined = np.concatenate([energy_ratios_R, energy_ratios_G, energy_ratios_B])
        energy_ratios_combined = energy_ratios_combined.reshape((1, -1))
        print(energy_ratios_combined.shape)
        assert energy_ratios_combined.shape[1] == 207  #K=84 
        all_energy_ratios.append(energy_ratios_combined)
        all_labels.append(1)  #clean label 1
        (B, G, R) = cv2.split(adversarial_images_numpy )
        fRadv = np.fft.fft2(R)
        fGadv = np.fft.fft2(G)
        fBadv = np.fft.fft2(B)
        fshiftRadv = np.fft.fftshift(fRadv)
        fshiftGadv = np.fft.fftshift(fGadv)
        fshiftBadv = np.fft.fftshift(fBadv)
        fadvimgR = np.log(np.abs(fshiftRadv))
        fadvimgG = np.log(np.abs(fshiftGadv))
        fadvimgB = np.log(np.abs(fshiftBadv))
        radius_range_adv, energy_advratios_R = energy(fadvimgR,K=84)
        radius_range_adv, energy_advratios_G = energy(fadvimgG,K=84)
        radius_range_adv, energy_advratios_B = energy(fadvimgB,K=84)
        max_length = max(len(energy_advratios_R), len(energy_advratios_G), len(energy_advratios_B))
        energy_advratios_R = np.pad(energy_advratios_R, (0, max_length - len(energy_advratios_R)))
        energy_advratios_G = np.pad(energy_advratios_G, (0, max_length - len(energy_advratios_G)))
        energy_advratios_B = np.pad(energy_advratios_B, (0, max_length - len(energy_advratios_B)))
        energy_advratios_combined = np.concatenate([energy_advratios_R, energy_advratios_G, energy_advratios_B])
        energy_advratios_combined = energy_advratios_combined.reshape((1, -1))
        print(energy_advratios_combined.shape)
        assert energy_ratios_combined.shape[1] == 207  #K=84
        all_energy_ratios.append(energy_advratios_combined)
        all_labels.append(0)  # adv label 0
energy_array = np.array(all_energy_ratios)
all_labels= np.array(all_labels)
with open('imagenetdataset.pkl', 'wb') as f:
    pickle.dump((energy_array, all_labels), f)
np.savez('imagenetdataset.npz', energy=energy_array, labels=all_labels)
##Generate a test set to test the REAR detector
class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
adv_loss =  nn.CrossEntropyLoss()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])  
data_dir = "./data"
data_clean(data_dir)
normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=20, shuffle=False)
all_energy_ratios_test = []
all_labels_test = []              
test.eval()
images, labels =  normal_data [0]
normal_iter = iter(normal_loader)
tar_cnt=1800  #1800
total_test_step=0
for i in range(tar_cnt//20):
    print("Iter: ", i)
    if total_test_step==90:
        break
    total_test_step+=1
    imgs, targets = next(normal_iter)
    imgs = imgs.detach().clone()
    targets = targets.detach().clone()
    imgs = imgs.to(device)
    targets = targets.to(device)
    imgs=imgs*255.0 
    # For target attack: set random target. 
    # Comment if you set untargeted attack.
    # labels = torch.from_numpy(np.random.randint(0, 1000, size = batch_size))
    attack = InfoDrop(test, batch_size=20, q_size =100, steps=50, targeted =False)
    at_images, labels, suc_step = attack(imgs, targets)   
    at_images = at_images.detach().clone()
    at_images =at_images.to(device)   
    clean_images = imgs    
    adversarial_images = at_images 
    clean_images_numpy = clean_images.detach().cpu().numpy()  
    adversarial_images_numpy = adversarial_images.detach().cpu().numpy() 
    clean_images_numpy = clean_images_numpy[0]
    imgs= np.moveaxis(clean_images_numpy, 0, 2)
    adv_img =  adversarial_images_numpy[0]
    at_images = np.moveaxis(adv_img, 0, 2)
    (B, G, R) = cv2.split(imgs)  
    fR = np.fft.fft2(R)
    fG = np.fft.fft2(G)
    fB = np.fft.fft2(B)
    fshiftR = np.fft.fftshift(fR)
    fshiftG = np.fft.fftshift(fG)
    fshiftB = np.fft.fftshift(fB)
    fimgR = np.log(np.abs(fshiftR))
    fimgG = np.log(np.abs(fshiftG))
    fimgB = np.log(np.abs(fshiftB))
    radius_range_clean, energy_ratios_R = energy(fimgR,K=84)
    radius_range_clean, energy_ratios_G = energy(fimgG,K=84)
    radius_range_clean, energy_ratios_B = energy(fimgB,K=84)
    max_length = max(len(energy_ratios_R), len(energy_ratios_G), len(energy_ratios_B))
    energy_ratios_R = np.pad(energy_ratios_R, (0, max_length - len(energy_ratios_R)))
    energy_ratios_G = np.pad(energy_ratios_G, (0, max_length - len(energy_ratios_G)))
    energy_ratios_B = np.pad(energy_ratios_B, (0, max_length - len(energy_ratios_B)))
    energy_ratios_combined = np.concatenate([energy_ratios_R, energy_ratios_G, energy_ratios_B])
    energy_ratios_combined = energy_ratios_combined.reshape((1, -1))
    assert energy_ratios_combined.shape[1] == 207  
    all_energy_ratios_test.append(energy_ratios_combined)
    all_labels_test.append(1)  # clean label 1
    (B, G, R) = cv2.split(at_images)
    fRadv = np.fft.fft2(R)
    fGadv = np.fft.fft2(G)
    fBadv = np.fft.fft2(B)
    fshiftRadv = np.fft.fftshift(fRadv)
    fshiftGadv = np.fft.fftshift(fGadv)
    fshiftBadv = np.fft.fftshift(fBadv)
    fadvimgR = np.log(np.abs(fshiftRadv))
    fadvimgG = np.log(np.abs(fshiftGadv))
    fadvimgB = np.log(np.abs(fshiftBadv))
    radius_range_adv, energy_advratios_R = energy(fadvimgR,K=84)
    radius_range_adv, energy_advratios_G = energy(fadvimgG,K=84)
    radius_range_adv, energy_advratios_B = energy(fadvimgB,K=84)
    max_length = max(len(energy_advratios_R), len(energy_advratios_G), len(energy_advratios_B))
    energy_advratios_R = np.pad(energy_advratios_R, (0, max_length - len(energy_advratios_R)))
    energy_advratios_G = np.pad(energy_advratios_G, (0, max_length - len(energy_advratios_G)))
    energy_advratios_B = np.pad(energy_advratios_B, (0, max_length - len(energy_advratios_B)))
    energy_advratios_combined = np.concatenate([energy_advratios_R, energy_advratios_G, energy_advratios_B])
    energy_advratios_combined = energy_advratios_combined.reshape((1, -1))
    assert energy_ratios_combined.shape[1] == 207  
    all_energy_ratios_test.append(energy_advratios_combined)
    all_labels_test.append(0)  # adv label 0
energy_array_test = np.array(all_energy_ratios_test)
all_labels_test = np.array(all_labels_test)
with open('imagenettest_dataset.pkl', 'wb') as f1:
    pickle.dump((energy_array_test, all_labels_test), f1)
np.savez('imagenettest_dataset.npz', energy=energy_array_test, labels=all_labels_test)

# training REAR 
with open('imagenetdataset.pkl', 'rb') as f:  
    energy_array, all_labels = pickle.load(f)
energy_tensor = torch.tensor(energy_array, dtype=torch.float32)
labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
batch_size = 32
dataset = TensorDataset(energy_tensor, labels_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
with open('imagenettest_dataset.pkl', 'rb') as f1:
    energy_array_test, all_labels_test = pickle.load(f1)
energy_tensor_test = torch.tensor(energy_array_test, dtype=torch.float32)
labels_tensor_test = torch.tensor(all_labels_test, dtype=torch.float32)
testdataset = TensorDataset(energy_tensor_test, labels_tensor_test)
batch_size = 32
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
testdataset_size = len(testdataset)

# REAR model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(128)  # 添加BatchNormalization
        # self.batchnorm2 = nn.BatchNorm1d(64)
        # self.batchnorm3 = nn.BatchNorm1d(32)
    def forward(self, x):
        x = x.view(-1, x.shape[2]) 
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)  
        return x
input_size = 207  #  the length of the energy ratio data selected for an image spectrum ; K=84  
model = BinaryClassifier(input_size)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001) 
# save REAR model
model_dir = "./model"
os.makedirs(model_dir, exist_ok=True)
num_epochs =50  
for epoch in range(num_epochs):
    model.train() 
    total_loss = 0.0
    for batch_data, batch_labels in loader:  
        optimizer.zero_grad()
        outputs = model(batch_data) 
        batch_labels = batch_labels.view(-1, 1).float()  
        loss = criterion(outputs, batch_labels)  
        loss.backward()  
        total_loss += loss.item() * batch_data.size(0)
        optimizer.step() 
    epoch_loss = total_loss / testdataset_size
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
    model_filename = os.path.join(model_dir, "BinaryClassifier_{}.pth".format(epoch))
    torch.save(model, model_filename)
    print("Model saved")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    model.eval()
    # Lists to store true labels and predicted probabilities
    true_labels = []
    pred_probs = []
    test_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_labels in testloader:
            outputs = model(batch_data)
            pred_probs.extend(outputs.numpy())
            true_labels.extend(batch_labels.numpy())
            batch_labels = batch_labels.view(-1, 1).float()
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item() * batch_data.size(0)
    test_epoch_loss = test_loss / testdataset_size
    print(f"Epoch [{epoch+1}/{num_epochs}] - Test Loss: {test_epoch_loss:.4f}")
    # Convert predicted probabilities to binary predictions
    predictions = [1 if prob >= 0.5 else 0 for prob in pred_probs]
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    save_results(epoch + 1, loss.item(), accuracy,precision,recall,f1)
    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    
###use the REAR 
# input_size =207 
# model = BinaryClassifier(input_size)
# list1=[0]
# model= torch.load("./BinaryClassifier_{}.pth")
# model.eval()

# def data_clean(data_dir):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     path_list=os.listdir(data_dir)
#     list=[]
#     lis=[]
#     global sum
#     sum=0
#     global k
#     k=0
#     import torch.optim as optim
#     import numpy as np 
#     import cv2
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#     true_labels = []
#     pred_probs = []
#     from collections import Counter
#     all_predictions = []
#     for img_name in path_list:
#         img_path = os.path.join(data_dir, img_name)
#         if not img_name.endswith(".npy"):
#                 os.remove(img_path)
#         img=np.load(img_path)
#         # print(img.shape)
#         new_size = (224, 224)
#         img = cv2.resize(img, new_size)
#         (B, G, R) = cv2.split(img)
#         fRadv = np.fft.fft2(R)
#         fGadv = np.fft.fft2(G)
#         fBadv = np.fft.fft2(B)
#         fshiftRadv = np.fft.fftshift(fRadv)
#         fshiftGadv = np.fft.fftshift(fGadv)
#         fshiftBadv = np.fft.fftshift(fBadv)
#         fadvimgR = np.log(np.abs(fshiftRadv))
#         fadvimgG = np.log(np.abs(fshiftGadv))
#         fadvimgB = np.log(np.abs(fshiftBadv))
#         radius_range_adv, energy_advratios_R = energy(fadvimgR,K=84)
#         radius_range_adv, energy_advratios_G = energy(fadvimgG,K=84)
#         radius_range_adv, energy_advratios_B = energy(fadvimgB,K=84)
#         max_length = max(len(energy_advratios_R), len(energy_advratios_G), len(energy_advratios_B))
#         energy_advratios_R = np.pad(energy_advratios_R, (0, max_length - len(energy_advratios_R)))
#         energy_advratios_G = np.pad(energy_advratios_G, (0, max_length - len(energy_advratios_G)))
#         energy_advratios_B = np.pad(energy_advratios_B, (0, max_length - len(energy_advratios_B)))
#         energy_advratios_combined = np.concatenate([energy_advratios_R, energy_advratios_G, energy_advratios_B])
#         energy_advratios_combined = energy_advratios_combined.reshape((1, -1))
#         # Lists to store true labels and predicted probabilities
#         true_labels = []
#         pred_probs = []
#         energy_advratios_combined= torch.tensor(energy_advratios_combined, dtype=torch.float32)
#         # Iterate through the dataset
#         with torch.no_grad():
#             # print(energy_advratios_combined.shape)
#             outputs = model(energy_advratios_combined)
#             pred_probs.extend(outputs.numpy())
#             true_labels.extend(list1)
#         # Convert predicted probabilities to binary predictions
#         predictions = [1 if prob >= 0.5 else 0 for prob in pred_probs]
#         all_predictions.extend(predictions)
#     predictions_counter = Counter(all_predictions)
#     num_0 = predictions_counter[0]
#     num_1 = predictions_counter[1]
#     return num_0,num_1
# data_dir="./adv_data"   
# num0,num1=data_clean(data_dir)  
# print(num0,num1)  