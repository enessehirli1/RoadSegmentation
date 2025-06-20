import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '.png'))
        
        # Debug: File path control
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Mask converting for 2 classes
        mask_processed = np.zeros_like(mask, dtype=np.int64)
        mask_processed[mask == 0] = 0      # Class 0 (Background)
        mask_processed[mask == 255] = 1    # Class 1 (Foreground)
        
        mask = mask_processed

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
