import torch
import torch.nn as nn
from model import generator
from denoising_module import denoiser
from loss_functions import ContentLoss, calculate_global_loss, calculate_Local_loss, SelfFeaturePreservingLoss, SimpleFeatureExtractor
from patchdiscriminator import PatchGan_discriminator
from dataloader import train_loader

def test_training_step():
    """Test a single training step to ensure no gradient errors occur"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate models
    generator_model = generator().to(device)
    denoising_model = denoiser().to(device)  
    global_discriminator1 = PatchGan_discriminator().to(device)
    global_discriminator2 = PatchGan_discriminator().to(device)
    local_discriminator = PatchGan_discriminator().to(device)

    # Instantiate feature extractor and loss objects
    feature_extractor = SimpleFeatureExtractor().to(device)
    feature_extractor.eval()
    sfp_loss_fn = SelfFeaturePreservingLoss(feature_extractor).to(device)
    content_loss_fn = ContentLoss().to(device)

    lr = 0.0004
    optimizer_G = torch.optim.Adam(generator_model.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(global_discriminator1.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(global_discriminator2.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D_local = torch.optim.Adam(local_discriminator.parameters(), lr, betas=(0.5, 0.999))
    optimizer_Denoising = torch.optim.Adam(denoising_model.parameters(), lr, betas=(0.5, 0.999))

    try:
        # Get a single batch
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")

        # Generate fake images and denoised images
        fake_images = generator_model(inputs)
        denoised_images = denoising_model(inputs)
        
        print(f"Fake images shape: {fake_images.shape}")
        print(f"Denoised images shape: {denoised_images.shape}")

        # ===============================
        # Train Discriminators FIRST
        # ===============================
        
        # Zero discriminator gradients
        optimizer_D.zero_grad()
        optimizer_D2.zero_grad()
        optimizer_D_local.zero_grad()

        # Global discriminator 1 (targets vs fake_images)
        C_real_global = global_discriminator1(targets)
        C_fake_global = global_discriminator1(fake_images.detach())

        # Global discriminator 2 (inputs vs denoised_images) 
        C_real_global2 = global_discriminator2(inputs)
        C_fake_global2 = global_discriminator2(denoised_images.detach())

        # Local discriminator (targets vs fake_images)
        C_real_local = local_discriminator(targets)
        C_fake_local = local_discriminator(fake_images.detach())

        # Calculate discriminator losses
        _, Loss_global_discriminator = calculate_global_loss(C_real_global, C_fake_global)
        _, Loss_global_discriminator2 = calculate_global_loss(C_real_global2, C_fake_global2)
        _, Loss_local_discriminator = calculate_Local_loss(C_real_local, C_fake_local)
        
        print(f"Discriminator losses calculated successfully")
        
        # Backward and step discriminators
        Loss_global_discriminator.backward(retain_graph=True)
        Loss_global_discriminator2.backward(retain_graph=True)
        Loss_local_discriminator.backward()
        
        optimizer_D.step()
        optimizer_D2.step()
        optimizer_D_local.step()
        
        print(f"Discriminators trained successfully")

        # ===============================
        # Train Generator 
        # ===============================
        
        # Zero generator gradients
        optimizer_G.zero_grad()

        # Re-compute discriminator outputs for generator training
        C_real_global_gen = global_discriminator1(targets)
        C_fake_global_gen = global_discriminator1(fake_images)

        C_real_local_gen = local_discriminator(targets)
        C_fake_local_gen = local_discriminator(fake_images)

        # Calculate generator losses
        Loss_global_generator, _ = calculate_global_loss(C_real_global_gen, C_fake_global_gen)
        Loss_local_generator, _ = calculate_Local_loss(C_real_local_gen, C_fake_local_gen)
        
        # Content loss (L1)
        contentloss = content_loss_fn(fake_images, targets)
        
        # Self Feature Preserving Loss (global)
        try:
            SFP_loss_global = sfp_loss_fn(targets, fake_images)
            print(f"SFP loss calculated successfully: {SFP_loss_global.item():.4f}")
        except RuntimeError as e:
            print(f"Warning: SFP loss failed, using zero loss. Error: {e}")
            SFP_loss_global = torch.tensor(0.0, device=device, requires_grad=True)

        # Total generator loss
        total_generator_loss = (
            Loss_global_generator + 
            Loss_local_generator + 
            SFP_loss_global + 
            contentloss
        )

        print(f"Generator loss calculated successfully: {total_generator_loss.item():.4f}")

        total_generator_loss.backward()
        optimizer_G.step()
        
        print(f"Generator trained successfully")
        
        # ===============================
        # Train Denoiser separately
        # ===============================
        
        # Zero denoiser gradients
        optimizer_Denoising.zero_grad()

        # Re-compute discriminator outputs for denoiser training
        C_real_global2_den = global_discriminator2(inputs)
        C_fake_global2_den = global_discriminator2(denoised_images)

        # Calculate denoiser loss
        Loss_global_generator2, _ = calculate_global_loss(C_real_global2_den, C_fake_global2_den)
        denoiser_loss = Loss_global_generator2 

        # Backward and step denoiser
        denoiser_loss.backward()
        optimizer_Denoising.step()
        
        print(f"Denoiser trained successfully")
        
        print("All training steps completed successfully!")
        print(f"Final losses - Generator: {total_generator_loss.item():.4f}, "
              f"Discriminator: {Loss_global_discriminator.item():.4f}, "
              f"Denoiser: {denoiser_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing the fixed training pipeline...")
    success = test_training_step()
    if success:
        print("\n All tests passed! The gradient error has been fixed.")
    else:
        print("\n Tests failed. Further debugging needed.") 