import torch
import torch.nn as nn
import time
from model import generator
from denoising_module import denoiser
from patchdiscriminator import PatchGan_discriminator

def test_performance():
    print("=== Test de Performance CPU vs GPU ===")
    
    # Créer un tenseur de test
    test_input = torch.randn(8, 1, 256, 320)
    
    # Test sur CPU
    print("\n--- Test sur CPU ---")
    device_cpu = torch.device("cpu")
    
    gen_cpu = generator().to(device_cpu)
    denoiser_cpu = denoiser().to(device_cpu)
    disc_cpu = PatchGan_discriminator().to(device_cpu)
    
    input_cpu = test_input.to(device_cpu)
    
    # Warmup
    for _ in range(3):
        _ = gen_cpu(input_cpu)
        _ = denoiser_cpu(input_cpu)
        _ = disc_cpu(input_cpu)
    
    # Test CPU
    start_time = time.time()
    for _ in range(10):
        fake_cpu = gen_cpu(input_cpu)
        denoised_cpu = denoiser_cpu(input_cpu)
        disc_out_cpu = disc_cpu(input_cpu)
    cpu_time = time.time() - start_time
    
    print(f"Temps CPU pour 10 itérations: {cpu_time:.4f} secondes")
    print(f"Temps CPU par itération: {cpu_time/10:.4f} secondes")
    
    # Test sur GPU
    if torch.cuda.is_available():
        print("\n--- Test sur GPU ---")
        device_gpu = torch.device("cuda")
        
        gen_gpu = generator().to(device_gpu)
        denoiser_gpu = denoiser().to(device_gpu)
        disc_gpu = PatchGan_discriminator().to(device_gpu)
        
        input_gpu = test_input.to(device_gpu)
        
        # Synchroniser GPU
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(3):
            _ = gen_gpu(input_gpu)
            _ = denoiser_gpu(input_gpu)
            _ = disc_gpu(input_gpu)
        
        torch.cuda.synchronize()
        
        # Test GPU
        start_time = time.time()
        for _ in range(10):
            fake_gpu = gen_gpu(input_gpu)
            denoised_gpu = denoiser_gpu(input_gpu)
            disc_out_gpu = disc_gpu(input_gpu)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"Temps GPU pour 10 itérations: {gpu_time:.4f} secondes")
        print(f"Temps GPU par itération: {gpu_time/10:.4f} secondes")
        
        # Comparaison
        speedup = cpu_time / gpu_time
        print(f"\n--- Comparaison ---")
        print(f"Accélération GPU: {speedup:.2f}x plus rapide")
        print(f"GPU est {speedup:.1f}x plus rapide que CPU")
        
        # Vérifier l'utilisation GPU
        print(f"\n--- Utilisation GPU ---")
        print(f"Mémoire GPU utilisée: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"Mémoire GPU réservée: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
        
        if speedup > 2:
            print("✅ GPU fonctionne correctement et accélère significativement!")
        else:
            print("⚠️ GPU fonctionne mais l'accélération est limitée")
            
    else:
        print("❌ CUDA non disponible!")

def test_current_training_speed():
    print("\n=== Test de vitesse d'entraînement actuel ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé: {device}")
    
    # Simuler un batch d'entraînement
    batch_size = 8
    test_input = torch.randn(batch_size, 1, 256, 320).to(device)
    test_target = torch.randn(batch_size, 1, 256, 320).to(device)
    
    # Modèles
    gen = generator().to(device)
    denoiser_model = denoiser().to(device)
    disc = PatchGan_discriminator().to(device)
    
    # Optimiseurs
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=0.0004)
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=0.0004)
    optimizer_Denoising = torch.optim.Adam(denoiser_model.parameters(), lr=0.0004)
    
    # Test d'entraînement complet
    start_time = time.time()
    
    # Forward pass
    fake_images = gen(test_input)
    denoised_images = denoiser_model(test_input)
    
    # Discriminator
    optimizer_D.zero_grad()
    C_real = disc(test_target)
    C_fake = disc(fake_images.detach())
    disc_loss = torch.mean((C_real - 1)**2 + C_fake**2)
    disc_loss.backward()
    optimizer_D.step()
    
    # Generator
    optimizer_G.zero_grad()
    C_fake_gen = disc(fake_images)
    gen_loss = torch.mean((C_fake_gen - 1)**2)
    gen_loss.backward()
    optimizer_G.step()
    
    # Denoiser
    optimizer_Denoising.zero_grad()
    denoiser_loss = torch.mean((denoised_images - test_input)**2)
    denoiser_loss.backward()
    optimizer_Denoising.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    print(f"Temps pour un batch complet: {total_time:.4f} secondes")
    print(f"Temps par batch: {total_time:.4f} secondes")
    
    # Estimation pour 2169 batches
    total_epoch_time = total_time * 2169
    print(f"Temps estimé par epoch: {total_epoch_time:.1f} secondes ({total_epoch_time/60:.1f} minutes)")
    
    if torch.cuda.is_available():
        print(f"Mémoire GPU: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"GPU utilisé: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    test_performance()
    test_current_training_speed() 