import torch
import torch.nn as nn
from model import generator
from denoising_module import denoiser
from patchdiscriminator import PatchGan_discriminator

def check_gpu_usage():
    print("=== Vérification GPU ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Test GPU memory
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("❌ CUDA non disponible!")
        return False
    
    print("\n=== Test des modèles sur GPU ===")
    
    # Créer un tenseur de test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor = torch.randn(2, 1, 64, 64).to(device)
    print(f"Test tensor device: {test_tensor.device}")
    
    # Tester les modèles
    try:
        # Générateur
        gen_model = generator().to(device)
        gen_output = gen_model(test_tensor)
        print(f"✅ Generator: {gen_output.device}, shape: {gen_output.shape}")
        
        # Débruiteur
        denoiser_model = denoiser().to(device)
        denoiser_output = denoiser_model(test_tensor)
        print(f"✅ Denoiser: {denoiser_output.device}, shape: {denoiser_output.shape}")
        
        # Discriminateur
        disc_model = PatchGan_discriminator().to(device)
        disc_output = disc_model(test_tensor)
        print(f"✅ Discriminator: {disc_output.device}, shape: {disc_output.shape}")
        
        # Vérifier la mémoire GPU après chargement des modèles
        print(f"\nGPU memory after loading models: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Test de performance simple
        print("\n=== Test de performance ===")
        import time
        
        start_time = time.time()
        for _ in range(10):
            _ = gen_model(test_tensor)
        end_time = time.time()
        
        print(f"Temps pour 10 forward passes: {end_time - start_time:.4f} secondes")
        
        # Vérifier que les paramètres sont sur GPU
        print("\n=== Vérification des paramètres ===")
        for name, param in gen_model.named_parameters():
            if param.device.type != 'cuda':
                print(f"❌ {name}: {param.device}")
            else:
                print(f"✅ {name}: {param.device}")
                break  # Juste le premier pour éviter trop de sortie
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    success = check_gpu_usage()
    if success:
        print("\n🎉 GPU fonctionne correctement!")
    else:
        print("\n💥 Problème avec le GPU!") 