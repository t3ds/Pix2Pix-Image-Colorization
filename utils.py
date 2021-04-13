import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from skimage import io, color
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2, use_dropout = False, use_bn = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size= 4, stride = stride, padding = 1, padding_mode = "reflect")
        self.activation = nn.LeakyReLU(0.2)

        if use_bn == True:
            self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn
        if use_dropout == True:
            self.drop = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.drop(x)
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, use_bn=False, stride = 2):
        super(DecodeBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding = 1)

        self.activation = nn.ReLU()

        if use_bn == True:
            self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

        if use_dropout == True:
            self.drop = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.up(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.drop(x)
        return x

class ColorDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.transform = transform
        

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.list_files[idx]
        img_path = os.path.join(self.root_dir, img_file)

        sample = Image.open(img_path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
            
        sample = np.asarray(sample).reshape(256,256,3)
        
        lab = (color.rgb2lab(sample) +128)/255
        lab = lab[:,:,1:3].transpose((2,0,1))
        #print(lab.shape)
        gray = color.rgb2gray(sample)
        
        sample = torch.from_numpy(lab)
        gray = torch.from_numpy(gray).unsqueeze(0)

        return sample.float(), gray.float()
