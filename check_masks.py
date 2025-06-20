import os
import numpy as np
from PIL import Image

def check_mask_values(mask_dir, num_samples=5):
    """Check unique values in mask images to determine model output channels."""
    mask_files = os.listdir(mask_dir)[:num_samples]
    
    print(f"Checking {len(mask_files)} mask files...")
    
    all_values = set()
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert("L"))
        
        unique_vals = np.unique(mask)
        all_values.update(unique_vals)
        
        print(f"{mask_file}: min={mask.min()}, max={mask.max()}, unique={unique_vals}")
    
    print(f"\nAll unique values across masks: {sorted(all_values)}")
    return sorted(all_values)

if __name__ == "__main__":
    # Check the directory for training masks
    train_mask_dir = "data/train_masks/"
    
    if os.path.exists(train_mask_dir):
        unique_values = check_mask_values(train_mask_dir)
        
        print(f"\nRecommendations:")
        print(f"- Number of unique values: {len(unique_values)}")
        print(f"- Model output channels should be: {len(unique_values)}")
        print(f"- If binary segmentation (0,255): use 2 output channels")
        print(f"- Map 255 -> 1 in dataset preprocessing")
    else:
        print(f"Directory {train_mask_dir} not found!")