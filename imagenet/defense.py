from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import os
import cv2
from torchvision import models      
from PIL import Image                 
from torchvision import transforms    
import json
import numpy as np
import time
import skimage
from skimage import io
import torchvision.transforms as transforms  
try:
    import cPickle as pickle
except ImportError:
    import pickle
import math
from torchvision.transforms import ToPILImage, ToTensor
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
import os.path
from math import sqrt, exp
from scipy import linalg
from skimage.restoration import denoise_nl_means
import io
# from mylib.utilities import myclip, mypsnr, conv_by_fft2
import matplotlib.pyplot as plt
import glob
from matplotlib import pyplot as plt
import scipy
import scipy.misc
from scipy import ndimage
from utils1 import *
class PlotData():
    def __init__(self, x_axis, y_axis, label = "", marker = ".", linestyle = "-"):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.label = label
        self.marker = marker
        self.linestyle = linestyle

class FigData():
    def __init__(self, x_label = "", x_fontsize = 18, x_scale = "linear"
                , y_label = "", y_fontsize = 18, y_scale = "linear"
                , savepath = None):
        self.x_label  = x_label
        self.y_label = y_label
        self.x_fontsize = x_fontsize
        self.y_fontsize = y_fontsize
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.savepath = savepath

def myplot(FigData, PlotDatas, plt_show = False):
    plt.figure()
    for p in PlotDatas:
        plt.plot(p.x_axis, p.y_axis, label=p.label, marker=p.marker, linestyle=p.linestyle)
    plt.xlabel(FigData.x_label, fontsize=FigData.x_fontsize)
    plt.ylabel(FigData.y_label, fontsize=FigData.y_fontsize)
    plt.xscale(FigData.x_scale)
    plt.yscale(FigData.y_scale)
    plt.grid() 
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.savefig(FigData.savepath, dpi = 200)  if FigData.savepath is not None else None
    plt.show() if plt_show is True else None

def myclip(x):
    return np.clip(x, 0, 255).astype(int)

def mypsnr(img, x):
    return peak_signal_noise_ratio(img, myclip(x), data_range = 255)
def conv_by_fft2(A, B, flag = 0, epsilon = 0, multichannel = 1):
    w, h = A.shape[0:2]
    wb,hb = B.shape[0:2]

    sz = (w - wb, h - hb)
    bigB = np.pad(B, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')
    bigB = fftpack.ifftshift(bigB)
    fft2B = fftpack.fft2(bigB)

    if multichannel == 1: # color
        if flag == 1:
            fft2B = fft2B.conjugate() / (abs(fft2B) **2 + epsilon*np.ones([w,h])) 
        C = np.empty(A.shape)
        for i in range(3):
            C[:,:,i] = np.real(fftpack.ifft2(fftpack.fft2(A[:,:,i]) * fft2B))
        return C

    else: # single channel
        if flag == 1:
            fft2B = fft2B.conjugate() / (abs(fft2B) **2 + epsilon*np.ones([w,h]))  
        return np.real(fftpack.ifft2(fftpack.fft2(A) * fft2B))      
def gaussian_kernel(ksize = 3):
    combs = [1]

    for i in range(1, ksize):
        ratio = (ksize-i)/(i)
        combs.append(combs[-1]*ratio)

    combs = np.array(combs).reshape(1,ksize)/(2**(ksize-1))
    return combs.T.dot(combs)
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())
def uint2single(img):
    return np.float32(img/255.)
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
def denoise_ffdnet(Y, sigma, color=1):
    if color == 1:
        model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    if color == 0:
        model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    show_img = True                     # default: False
    # show_img = False                     # default: False
    task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    sf = 1                    # unused for denoising
    if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
    else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
    model_pool = './ffdnet/ffdnet-main/pnp/model_zoo' 
    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    img_L = uint2single(Y)
    img_L = single2tensor4(img_L)
    img_L = img_L.to(device)
    sigma = torch.full((1,1,1,1), sigma/255.).type_as(img_L)
    img_E = model(img_L, sigma)
    X = tensor2uint(img_E) 
    return X
def estimate_noise(img, block_size=8):
    h, w = img.shape[:2]
    noise_var = np.zeros((h // block_size, w // block_size), np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            noise_var[i//block_size, j//block_size] = np.var(block)
    img_noise_var = np.mean(noise_var)
    img_noise_var = (img_noise_var / np.max(noise_var)) *75
    if img_noise_var<15:
        img_noise_var=15
    return img_noise_var
data_dir="./adv_samples"  
list1 =[]
list2 =[]
def data_adv(data_dir):
    global z
    global summ
    summ=0
    z=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_list=os.listdir(data_dir)
    for img_name in path_list:
        img_path = os.path.join(data_dir, img_name)
        if not img_name.endswith(".npy"):
                os.remove(img_path)
        img=np.load(image_path) 
        img=Image.fromarray(np.uint8(img))
        resize = transforms.Resize([224,224])
        img = resize(img)
        img=np.array(img)
         # img=io.imread(image_path)
        img=img/255.0
        var1=estimate_noise(img, block_size=8)
        var1=var1/255.0
        var1=var1*var1
        img1=img
        dst = skimage.util.random_noise(img1, mode='gaussian', var=var1)
        img1=dst
        noisy_img=img1.astype(np.float32) 
        img=noisy_img
        sigma=int(np.sqrt(var1)*255.0)
        if img.max() <= 1: img*= 255
        img = denoise_ffdnet(img, sigma)
        preprocess = transforms.Compose([      
        #         transforms.Resize((224, 224)),  
                transforms.ToTensor(),          
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   
                std=[0.229, 0.224, 0.225]
         )])
        img=Image.fromarray(np.uint8(img))
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).float().to(device)    
        # print(img_t.size()  )  torch.Size([3, 224, 224])
        # print(batch_t.size()  )  torch.Size([1, 3, 224, 224])
        resnet = models.resnet50(pretrained=True).to(device) 
        resnet.eval()           
        out = resnet(batch_t)
        out.size()
        with open('./imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        # _, indices = torch.sort(out, descending=True)
        # [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
        a=(classes[index[0]],round(percentage[index[0]].item(),6))
        a2=classes[index[0]]
        print(a2)
data_adv(data_dir)