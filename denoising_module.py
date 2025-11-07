import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
import numpy as np
import os
import random

def apply_high_contrast_and_noise(image):

    img = image.float()  
    B, C, H, W = img.shape

    # Define sharpening kernel
    kernel = torch.tensor([[[[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]]]], dtype=torch.float32, device=img.device)
    kernel = kernel.repeat(C, 1, 1, 1)  # shape: [C, 1, 3, 3]

    # Apply sharpening using grouped conv
    contrast_img = F.conv2d(img, kernel, padding=1, groups=C)
    contrast_img = contrast_img.clamp(0, 255)

    # Add Gaussian noise with per-batch std
    noisy_img = torch.empty_like(contrast_img)
    for b in range(B):
        std = random.uniform(0, 50)
        noise = torch.randn_like(contrast_img[b]) * std
        noisy_img[b] = (contrast_img[b] + noise).clamp(0, 255)

    return noisy_img.type_as(image)



class denoiser(nn.Module):
    def __init__(self, C_in=1, C_out=1):
        super(denoiser, self).__init__()
        self.convfirst = nn.Conv2d(C_in, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.ReLU1 = nn.ReLU(inplace=True)  
        self.convlayer = self.conv_layer(64, 64)
        self.convlast = nn.Conv2d(64, C_out, kernel_size=3, stride=1, padding=1, bias=True)

    def conv_layer(self, C_in, C_out):
        return nn.Sequential(
            nn.Conv2d(C_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(64),
            nn.Conv2d(64, C_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        x_processed = apply_high_contrast_and_noise(x)
        d1 = self.convfirst(x_processed)
        d2 = self.ReLU1(d1)
        d3 = self.convlayer(d2)
        d4 = self.convlast(d3)
        d5 = self.ReLU1(d4)
        d5 = d5 + x 
        return d5