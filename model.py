import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
import numpy as np
import os


def compute_attention_map(tensor):
    if not tensor.dtype.is_floating_point:
        tensor = tensor.float()
    
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    
    if tensor_min < 0 or tensor_max > 1:
        I_normalized = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    else:
        I_normalized = tensor
    
    attention_map = 1.0 - I_normalized
    
    return attention_map

class generator(nn.Module):
    def __init__(self, C_in = 1 , C_out = 1 ):
        super(generator, self).__init__()
        
# definition of the architecture of the generator model  : 

        self.down1 = self.conblock(C_in , 32)
        self.down2 = self.conblock(32 , 64)
        self.down3 = self.conblock(64, 128)
        self.down4 = self.conblock(128 , 256)

        self.up1 = self.conblock(256 + 256, 128)  # using the concatination with the attention map we somme the 2  components 
        self.up2 = self.conblock(128 + 128, 64)
        self.up3 = self.conblock(64 + 64, 32)
        self.up4 = self.conblock(32 + 32, C_out)

# definition of the convolutional block  :  

    def conblock(self, C_in1, C_out1):
        return nn.Sequential(
            nn.Conv2d(C_in1, C_out1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),  # Changed!
            nn.BatchNorm2d(C_out1),
            nn.Conv2d(C_out1, C_out1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),  # Changed!
            nn.BatchNorm2d(C_out1),
        )


    def forward(self, x):

# SIMPLE  downsampling method 

        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        d4 = self.down4(F.max_pool2d(d3, 2))
        d4_final = F.max_pool2d(d4, 2)


# building attention maps for  layer output 



        d4_attention = compute_attention_map(d4)
        d3_attention = compute_attention_map(d3)
        d2_attention = compute_attention_map(d2)
        d1_attention = compute_attention_map(d1)
                 
                 
# upsampling the feature maps using the method described which is  : interpolate with bilinear 
# then get the attention map and concatinate the interpolated map with it 
# then make a convolution to upsample it  
      
        u4 = F.interpolate(d4_final, scale_factor=2, mode='bilinear', align_corners=True)
        U4 = torch.cat([u4, d4_attention], dim=1)
        U4_a = self.up1(U4)
        
        u3 = F.interpolate(U4_a, scale_factor=2, mode='bilinear', align_corners=True)
        U3 = torch.cat([u3, d3_attention], dim=1)
        U3_a = self.up2(U3)
        
        u2 = F.interpolate(U3_a, scale_factor=2, mode='bilinear', align_corners=True)
        U2 = torch.cat([u2, d2_attention], dim=1)
        U2_a = self.up3(U2)
        
        u1 = F.interpolate(U2_a, scale_factor=2, mode='bilinear', align_corners=True)
        U1 = torch.cat([u1, d1_attention], dim=1)
        U1_a = self.up4(U1)

        big_attention_map = compute_attention_map(x)
        attentioned_outcome = torch.mul(big_attention_map,U1_a)

        final_result = attentioned_outcome + x
        
        return  final_result 