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
        
        # Get valid images from the image directory
        self.images = self.get_valid_images(image_dir)
        print(f"Found {len(self.images)} valid images in {image_dir}")

    def get_valid_images(self, directory):
        """Filter only valid image files in the directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        valid_images = []
        
        all_files = os.listdir(directory)
        print(f"Total files in directory: {len(all_files)}")
        
        for filename in all_files:
            # jump over hidden files and git files
            if filename.startswith('.'):
                print(f"Skipping hidden/git file: {filename}")
                continue
                
            # Check file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in valid_extensions:
                print(f"Skipping non-image file: {filename}")
                continue
                
            # Check if the file is a valid image
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  
                valid_images.append(filename)
            except Exception as e:
                print(f"Skipping corrupted file: {filename} - Error: {e}")
                
        return sorted(valid_images)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        
        # Change mask filename to match image filename
        mask_filename = os.path.splitext(img_filename)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Debug: File path control
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        try:
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
            
        except Exception as e:
            print(f"Error loading image/mask at index {index}: {e}")
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")
            return self.__getitem__((index + 1) % len(self.images))