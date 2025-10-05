import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

class StegoDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transform = transforms
        
        all_files = os.listdir(image_dir)
        self.image_paths = [os.path.join(image_dir, f) for f in all_files 
                           if f.lower().endswith(('.png'))]
            
        print(f"Found {len(self.image_paths)} valid image files in {image_dir}")

    def __getitem__(self, index):
        if index >= len(self.image_paths):
            raise IndexError(f"Index {index} out of range for dataset with {len(self.image_paths)} elements")
            
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return 0 as a dummy class label

    def __len__(self):
        return len(self.image_paths)

def get_transforms():
    """Get training transforms"""
    return transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

def create_dataloader(config):
    """Create training dataloader"""
    transform_train = get_transforms()
    train_dataset = StegoDataset(config.train_dir, transform_train)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader
