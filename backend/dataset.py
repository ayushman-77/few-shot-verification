import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os

class SiameseDataset(Dataset):
    """
    Custom Dataset for Siamese Network training.
    Yields pairs of images and a label indicating whether they belong to the same class (0) or different classes (1).
    """
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.classes = os.listdir(image_folder)
        self.class_to_images = {
            cls: [os.path.join(image_folder, cls, img) for img in os.listdir(os.path.join(image_folder, cls))]
            for cls in self.classes
        }
        self.classes_multiple = [cls for cls in self.classes if len(self.class_to_images[cls]) >= 2]

    def __len__(self):
        return sum([len(imgs) for imgs in self.class_to_images.values()])

    def __getitem__(self, index):
        same_class = random.choice([True, False])
        
        if same_class:
            cls = random.choice(self.classes_multiple)
            img1_path, img2_path = random.sample(self.class_to_images[cls], 2)
            label = torch.tensor(0, dtype=torch.float32)
        else:
            cls1, cls2 = random.sample(self.classes, 2)
            img1_path = random.choice(self.class_to_images[cls1])
            img2_path = random.choice(self.class_to_images[cls2])
            label = torch.tensor(1, dtype=torch.float32)
            
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label

def get_dataloader(image_folder, batch_size, transform=None):
    """
    Helper function to initialize the DataLoader.
    """
    dataset = SiameseDataset(image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
