import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, generated_image, ground_truth_image):
        return self.l1(generated_image, ground_truth_image)
    
def self_feature_preserving_patch_loss(real_img, gen_img, feature_extractor, patch_size=64, num_patches=5):
    """
    Patch-based feature preserving loss.
    Note: This function expects batch_size=1 for patch extraction.
    """
    B, C, H, W = real_img.shape
    assert B == 1, "This function expects a single image per batch for patch extraction."
    
    # Ensure patch size is valid
    if H < patch_size or W < patch_size:
        # If image is smaller than patch, use the whole image
        patch_size = min(H, W)
    
    losses = []
    device = real_img.device
    
    for _ in range(num_patches):
        top = torch.randint(0, H - patch_size + 1, (1,), device=device).item()
        left = torch.randint(0, W - patch_size + 1, (1,), device=device).item()
        
        real_patch = real_img[:, :, top:top+patch_size, left:left+patch_size]
        gen_patch = gen_img[:, :, top:top+patch_size, left:left+patch_size]
        
        real_feats = feature_extractor(real_patch)
        gen_feats = feature_extractor(gen_patch)
        
        # Accumulate patch losses in a list to avoid in-place ops
        patch_losses = [
            F.mse_loss(rf, gf)
            for rf, gf in zip(real_feats, gen_feats)
        ]
        patch_loss = torch.stack(patch_losses).mean()
        losses.append(patch_loss)
    
    return torch.stack(losses).mean()

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layer_ids=[(1,2), (2,1), (3,3)], use_instancenorm=False):
        super().__init__()
        from torchvision.models import VGG16_Weights
        vgg_pretrained = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        if isinstance(layer_ids, torch.Tensor):
            layer_ids = [tuple(x) for x in layer_ids.tolist()]
        
        self.layer_ids = layer_ids
        self.slices = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        current_pool = 0
        current_conv = 0
        selected_indexes = []
        conv_out_channels = []
        
        for idx, layer in enumerate(vgg_pretrained):
            if isinstance(layer, nn.MaxPool2d):
                current_pool += 1
                current_conv = 0
            elif isinstance(layer, nn.Conv2d):
                current_conv += 1
                
                # Check if this conv layer matches our target
                if (current_pool, current_conv) in self.layer_ids:
                    selected_indexes.append(idx)
                    conv_out_channels.append(layer.out_channels)
        
        prev_idx = 0
        for idx, out_channels in zip(selected_indexes, conv_out_channels):
            self.slices.append(nn.Sequential(*vgg_pretrained[prev_idx:idx+1]))
            if use_instancenorm:
                self.norms.append(nn.InstanceNorm2d(out_channels, affine=False))
            else:
                self.norms.append(nn.Identity())
            prev_idx = idx+1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:  # If input has 1 channel (grayscale)
            x = x.repeat(1, 3, 1, 1)  # Repeat the channel 3 times to make it RGB
        
        feats = []
        current_x = x
        for s, norm in zip(self.slices, self.norms):
            current_x = s(current_x)
            # Create a new tensor to avoid in-place operations
            normalized_feat = norm(current_x.clone())
            feats.append(normalized_feat)
        return feats

class SimpleFeatureExtractor(nn.Module):
    """
    A simplified feature extractor that avoids InstanceNorm issues
    """
    def __init__(self):
        super().__init__()
        # Define each block separately for correct chaining
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        return [feat1, feat2, feat3]

class SelfFeaturePreservingLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.criterion = nn.MSELoss()
        
    def forward(self, real_img, gen_img):
        # Ensure we're not modifying the input tensors
        real_img_clone = real_img.clone()
        gen_img_clone = gen_img.clone()
        
        real_feats = self.feature_extractor(real_img_clone)
        gen_feats = self.feature_extractor(gen_img_clone)
        
        # Accumulate losses in a list to avoid in-place ops!
        losses = []
        for rf, gf in zip(real_feats, gen_feats):
            loss = self.criterion(rf, gf)
            losses.append(loss)
        
        return torch.stack(losses).mean()

def calculate_global_loss(C_real, C_fake):
    """
    Calculate global adversarial loss using relativistic average discriminator.
    """
    # Ensure tensors are on the same device
    device = C_real.device
    
    # Calculate means
    mean_real = torch.mean(C_real)
    mean_fake = torch.mean(C_fake)
    
    # Relativistic discriminator outputs
    Dra1 = torch.sigmoid(C_real - mean_fake)
    Dra2 = torch.sigmoid(C_fake - mean_real)
    
    # Discriminator loss
    Loss_global_discriminator = torch.mean((Dra1 - 1)**2) + torch.mean(Dra2**2)
    
    # Generator loss
    Loss_global_generator = torch.mean(Dra1**2) + torch.mean((Dra2 - 1)**2)
    
    return Loss_global_generator, Loss_global_discriminator 

def calculate_Local_loss(C_real, C_fake):
    """
    Calculate local adversarial loss by sampling random locations.
    """
    batch, ch, h, w = C_real.shape
    device = C_real.device
    
    # Ensure we have valid dimensions
    if h == 0 or w == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    num_samples = min(5, batch * h * w)  # Don't sample more than available
    
    # Generate random indices
    batch_indices = torch.randint(0, batch, (num_samples,), device=device)
    h_indices = torch.randint(0, h, (num_samples,), device=device)
    w_indices = torch.randint(0, w, (num_samples,), device=device)
    
    # Sample from the feature maps
    # Assuming the discriminator output has shape [batch, channels, h, w]
    # and we want to sample from the first channel
    channel_idx = 0 if ch > 0 else 0
    
    real_samples = C_real[batch_indices, channel_idx, h_indices, w_indices]  
    fake_samples = C_fake[batch_indices, channel_idx, h_indices, w_indices]
    
    # Calculate losses
    Loss_local_discriminator = torch.mean((real_samples - 1)**2 + fake_samples**2)
    Loss_local_generator = torch.mean((fake_samples - 1)**2)
    
    return Loss_local_generator, Loss_local_discriminator