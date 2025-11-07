import torch
import torch.nn as nn
from model import generator
from denoising_module import denoiser
from loss_functions import ContentLoss, calculate_global_loss, calculate_Local_loss, SelfFeaturePreservingLoss, VGG16FeatureExtractor, SimpleFeatureExtractor, self_feature_preserving_patch_loss
from patchdiscriminator import PatchGan_discriminator
import torch.optim as optim
import os
from dataloader import train_loader
from torchvision.utils import save_image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Configuration flags
USE_SFP_LOSS = True  # Set to False to disable Self Feature Preserving Loss

# Optimisations GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = False  

# Instantiate models
print("Initializing models...")
generator_model = generator().to(device)
denoising_model = denoiser().to(device)  
global_discriminator1 = PatchGan_discriminator().to(device)
global_discriminator2 = PatchGan_discriminator().to(device)
local_discriminator = PatchGan_discriminator().to(device)

# Instantiate feature extractor and loss objects ONCE
if USE_SFP_LOSS:
    print("Initializing feature extractor...")
    feature_extractor = SimpleFeatureExtractor().to(device)
    feature_extractor.eval()  # Use eval mode for perceptual loss
    # Don't freeze parameters if you want gradients to flow
    for param in feature_extractor.parameters():
        param.requires_grad = False  # Freeze feature extractor
    sfp_loss_fn = SelfFeaturePreservingLoss(feature_extractor).to(device)
else:
    sfp_loss_fn = None

content_loss_fn = ContentLoss().to(device)

# Use different learning rates - discriminators should be slower
lr_gen = 0.0002
lr_disc = 0.0001  # Lower learning rate for discriminators

optimizer_G = torch.optim.Adam(generator_model.parameters(), lr_gen, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(global_discriminator1.parameters(), lr_disc, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(global_discriminator2.parameters(), lr_disc, betas=(0.5, 0.999))
optimizer_D_local = torch.optim.Adam(local_discriminator.parameters(), lr_disc, betas=(0.5, 0.999))
optimizer_Denoising = torch.optim.Adam(denoising_model.parameters(), lr_gen, betas=(0.5, 0.999))

# Load pretrained weights
generator_model.load_state_dict(torch.load('C:/Users/dhiaa/Desktop/project5/saved_weights/generator_epoch_61.pth'))
denoising_model.load_state_dict(torch.load('C:/Users/dhiaa/Desktop/project5/saved_weights/denoiser_epoch_61.pth'))
global_discriminator1.load_state_dict(torch.load('C:/Users/dhiaa/Desktop/project5/saved_weights/global_discriminator1_epoch_61.pth'))
global_discriminator2.load_state_dict(torch.load('C:/Users/dhiaa/Desktop/project5/saved_weights/global_discriminator2_epoch_61.pth'))
local_discriminator.load_state_dict(torch.load('C:/Users/dhiaa/Desktop/project5/saved_weights/local_discriminator_epoch_61.pth'))

num_epochs = 200
print(f"Starting training for {num_epochs} epochs...")
print(f"Total batches per epoch: {len(train_loader)}")

# Create directories for saving samples and weights
if not os.path.exists('samples'):
    os.makedirs('samples')
if not os.path.exists('C:/Users/dhiaa/Desktop/project5/saved_weights'):
    os.makedirs('C:/Users/dhiaa/Desktop/project5/saved_weights')

# Get a fixed sample (single image) for consistent visualization across epochs
print("Preparing fixed sample for visualization...")
with torch.no_grad():
    fixed_sample_inputs, fixed_sample_targets = next(iter(train_loader))
    fixed_sample_input = fixed_sample_inputs[0:1].to(device) 
    fixed_sample_target = fixed_sample_targets[0:1].to(device)
    print("Fixed sample prepared")

# Variables pour le monitoring
total_start_time = time.time()
epoch_times = []

# Loss weights for balancing
lambda_content = 10.0
lambda_sfp = 5.0  # Increase SFP weight
lambda_adv = 1.0

# Label smoothing to help training stability
real_label = 0.9  # Instead of 1.0
fake_label = 0.1  # Instead of 0.0

for epoch in range(60, 200):
    epoch_start_time = time.time()
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    
    # Training loop for each epoch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = inputs.size(0)

        # ===============================
        # Train Discriminators FIRST
        # ===============================
        
        # Generate fake images
        with torch.no_grad():
            fake_images = generator_model(inputs)
            denoised_images = denoising_model(inputs)

        # Train Global Discriminator 1 (real targets vs fake generated)
        optimizer_D.zero_grad()
        
        # Real samples
        real_pred_global = global_discriminator1(targets)
        real_labels = torch.full_like(real_pred_global, real_label, device=device)
        real_loss_global = nn.MSELoss()(real_pred_global, real_labels)
        
        # Fake samples
        fake_pred_global = global_discriminator1(fake_images.detach())
        fake_labels = torch.full_like(fake_pred_global, fake_label, device=device)
        fake_loss_global = nn.MSELoss()(fake_pred_global, fake_labels)
        
        Loss_global_discriminator = (real_loss_global + fake_loss_global) * 0.5
        Loss_global_discriminator.backward()
        optimizer_D.step()

        # Train Global Discriminator 2 (real inputs vs fake denoised)
        optimizer_D2.zero_grad()
        
        # Real samples (original inputs)
        real_pred_global2 = global_discriminator2(inputs)
        real_labels2 = torch.full_like(real_pred_global2, real_label, device=device)
        real_loss_global2 = nn.MSELoss()(real_pred_global2, real_labels2)
        
        # Fake samples (denoised images should look like inputs)
        fake_pred_global2 = global_discriminator2(denoised_images.detach())
        fake_labels2 = torch.full_like(fake_pred_global2, fake_label, device=device)
        fake_loss_global2 = nn.MSELoss()(fake_pred_global2, fake_labels2)
        
        Loss_global_discriminator2 = (real_loss_global2 + fake_loss_global2) * 0.5
        Loss_global_discriminator2.backward()
        optimizer_D2.step()

        # Train Local Discriminator (real targets vs fake generated)
        optimizer_D_local.zero_grad()
        
        # Real samples
        real_pred_local = local_discriminator(targets)
        real_labels_local = torch.full_like(real_pred_local, real_label, device=device)
        real_loss_local = nn.MSELoss()(real_pred_local, real_labels_local)
        
        # Fake samples
        fake_pred_local = local_discriminator(fake_images.detach())
        fake_labels_local = torch.full_like(fake_pred_local, fake_label, device=device)
        fake_loss_local = nn.MSELoss()(fake_pred_local, fake_labels_local)
        
        Loss_local_discriminator = (real_loss_local + fake_loss_local) * 0.5
        Loss_local_discriminator.backward()
        optimizer_D_local.step()
        
        optimizer_G.zero_grad()
        fake_images = generator_model(inputs)
        fake_pred_global_gen = global_discriminator1(fake_images)
        fake_pred_local_gen = local_discriminator(fake_images)
        
        gen_labels_global = torch.full_like(fake_pred_global_gen, real_label, device=device)
        gen_labels_local = torch.full_like(fake_pred_local_gen, real_label, device=device)
        
        Loss_global_generator = nn.MSELoss()(fake_pred_global_gen, gen_labels_global)
        Loss_local_generator = nn.MSELoss()(fake_pred_local_gen, gen_labels_local)
        
        # Content loss (L1)
        contentloss = content_loss_fn(fake_images, targets)
        
        # SFP Loss - always calculate if enabled
        if USE_SFP_LOSS and sfp_loss_fn is not None:
            SFP_loss_global = sfp_loss_fn(targets, fake_images)
            # Debug: print some statistics about SFP loss
            if batch_idx % 50 == 0:
                print(f"    SFP Loss debug - Raw value: {SFP_loss_global.item():.6f}")
        else:
            SFP_loss_global = torch.tensor(0.0, device=device)

        # Total generator loss with proper weighting
        total_generator_loss = (
            lambda_adv * (Loss_global_generator + Loss_local_generator) +
            lambda_content * contentloss +
            lambda_sfp * SFP_loss_global
        )

        total_generator_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(generator_model.parameters(), max_norm=1.0)
        
        optimizer_G.step()
        
        # ===============================
        # Train Denoiser separately
        # ===============================
        
        optimizer_Denoising.zero_grad()
        denoised_images = denoising_model(inputs)
        fake_pred_denoiser = global_discriminator2(denoised_images)
        denoiser_labels = torch.full_like(fake_pred_denoiser, real_label, device=device)
        denoiser_adv_loss = nn.MSELoss()(fake_pred_denoiser, denoiser_labels)
        denoiser_content_loss = content_loss_fn(denoised_images, inputs)
        denoiser_loss = lambda_adv * denoiser_adv_loss + lambda_content * denoiser_content_loss
        denoiser_loss.backward()
        torch.nn.utils.clip_grad_norm_(denoising_model.parameters(), max_norm=1.0)
        optimizer_Denoising.step()


        if batch_idx % 10 == 0:
            batch_time = time.time() - batch_start_time
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
            adv_loss_total = (Loss_global_generator + Loss_local_generator).item()
            content_loss_val = contentloss.item()
            sfp_loss_val = SFP_loss_global.item()

            print(f"[Epoch {epoch+1:03d} | Batch {batch_idx:04d}/{len(train_loader)} | {batch_time:.2f}s] "
                f"G_Loss: {total_generator_loss.item():.4f} (Adv: {adv_loss_total:.4f}, Content: {content_loss_val:.4f}, SFP: {sfp_loss_val:.6f}) | "
                f"D_Loss: G1: {Loss_global_discriminator.item():.4f}, G2: {Loss_global_discriminator2.item():.4f}, Local: {Loss_local_discriminator.item():.4f} | "
                f"Denoiser_Loss: {denoiser_loss.item():.4f} (Adv: {denoiser_adv_loss.item():.4f}, Content: {denoiser_content_loss.item():.4f}) | "
                f"GPU: {gpu_memory:.1f}MB")

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    real_mean = torch.mean(real_pred_global).item()
                    fake_mean = torch.mean(fake_pred_global).item()
                    print(f"    D1 predictions - Real: {real_mean:.4f}, Fake: {fake_mean:.4f}")

    print(f"Generating and saving sample images for epoch {epoch+1}...")
    with torch.no_grad():
        generator_model.eval()
        denoising_model.eval()
        
        fixed_fake = generator_model(fixed_sample_input)
        fixed_denoised = denoising_model(fixed_sample_input)
        
        comparison = torch.cat([fixed_sample_input, fixed_fake, fixed_sample_target], dim=0)
        save_image(comparison, f'samples/comparison_epoch_{epoch+1:03d}.png', normalize=True, nrow=1)
        save_image(fixed_fake, f'samples/generated_epoch_{epoch+1:03d}.png', normalize=True)
        save_image(fixed_denoised, f'samples/denoised_epoch_{epoch+1:03d}.png', normalize=True)
        
        generator_model.train()
        denoising_model.train()
        
        print(f"Sample images saved for epoch {epoch+1}")
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s (avg: {avg_epoch_time:.2f}s)")
    
    # Estimate remaining time
    remaining_epochs = num_epochs - (epoch + 1)
    estimated_time_remaining = remaining_epochs * avg_epoch_time
    print(f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes")

    # Save model weights at the end of each epoch
    torch.save(generator_model.state_dict(), f'C:/Users/dhiaa/Desktop/project5/saved_weights/generator_epoch_{epoch+1}.pth')
    torch.save(denoising_model.state_dict(), f'C:/Users/dhiaa/Desktop/project5/saved_weights/denoiser_epoch_{epoch+1}.pth')
    torch.save(global_discriminator1.state_dict(), f'C:/Users/dhiaa/Desktop/project5/saved_weights/global_discriminator1_epoch_{epoch+1}.pth')
    torch.save(global_discriminator2.state_dict(), f'C:/Users/dhiaa/Desktop/project5/saved_weights/global_discriminator2_epoch_{epoch+1}.pth')
    torch.save(local_discriminator.state_dict(), f'C:/Users/dhiaa/Desktop/project5/saved_weights/local_discriminator_epoch_{epoch+1}.pth')
    print(f"Saved model weights for epoch {epoch+1}")

total_time = time.time() - total_start_time
print(f"\nTraining completed in {total_time/60:.1f} minutes!")
print(f"Average time per epoch: {total_time/num_epochs:.2f} seconds")
print("\nSaved files:")
print("- Generated images per epoch: samples/generated_epoch_XXX.png")
print("- Denoised images per epoch: samples/denoised_epoch_XXX.png")
print("- Comparison images per epoch: samples/comparison_epoch_XXX.png")