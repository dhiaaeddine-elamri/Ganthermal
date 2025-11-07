from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class ThermalDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        
        self.filenames = os.listdir(input_dir)
        self.filenames = [f for f in self.filenames if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])
        
        input_image = Image.open(input_path).convert("L") 
        target_image = Image.open(target_path).convert("L")
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

transform = transforms.Compose([
    transforms.Resize((320, 256)),   
    transforms.ToTensor(),
])

dataset = ThermalDataset(r'C:\Users\dhiaa\Desktop\project5\ai_enova\blurred', r'C:\Users\dhiaa\Desktop\project5\ai_enova\TIR',  transform=transform)

from torch.utils.data import DataLoader

def create_train_loader(batch_size=8, num_workers=0):
    """
    Créer le DataLoader de manière sécurisée pour Windows
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 0 pour éviter les problèmes de multiprocessing
        pin_memory=True,  # Optimisation pour GPU
        persistent_workers=False  # Désactivé pour éviter les problèmes
    )

# Créer le train_loader par défaut avec num_workers=0 pour éviter les problèmes
train_loader = create_train_loader(batch_size=8, num_workers=0)