import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import RoadDataset  # Importing the RoadDataset from the dataset module
import os
import torch.nn.functional as F
import copy

"""
# Orijinal dataset import'u (aynı dosyayı kullanacağız)
import sys
sys.path.append('..')  # Bir üst klasörden import için
from dataset import RoadDataset
"""
def safe_collate_fn(batch):
    """
    Custom collate function to handle potential size mismatches
    """
    images, masks = zip(*batch)
    
    # Get the first image size as reference
    if len(images) > 0:
        ref_shape = images[0].shape[-2:]  # (H, W)
    else:
        return torch.tensor([]), torch.tensor([])
    
    # Check if all images have the same size
    all_same_size = all(img.shape[-2:] == ref_shape for img in images)
    
    if all_same_size:
        # If all same size, use default collate
        return torch.stack(images), torch.stack(masks)
    else:
        # If different sizes, resize to reference size
        # print(f"Warning: Found different image sizes in batch. Resizing to {ref_shape}")
        
        safe_images = []
        safe_masks = []
        
        for img, mask in zip(images, masks):
            if img.shape[-2:] != ref_shape:
                # Resize image
                img = F.interpolate(
                    img.unsqueeze(0), 
                    size=ref_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            if mask.shape[-2:] != ref_shape:
                # Resize mask (use nearest neighbor for masks)
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(), 
                    size=ref_shape, 
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            
            safe_images.append(img)
            safe_masks.append(mask)
        
        return torch.stack(safe_images), torch.stack(safe_masks)

def save_checkpoint(state, filename="best_model.pth.tar"):
    """
    Safe checkpoint saving that handles pickle issues
    """
    print("=> Saving checkpoint")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        # Create a safe copy of the state dictionary
        safe_state = {}
        
        # Copy non-problematic items
        for key, value in state.items():
            if key == 'state_dict':
                # Handle model state_dict specially
                try:
                    # Try to create a clean state dict
                    safe_state_dict = {}
                    for param_name, param_value in value.items():
                        if isinstance(param_value, torch.Tensor):
                            safe_state_dict[param_name] = param_value.detach().clone()
                        else:
                            safe_state_dict[param_name] = param_value
                    safe_state[key] = safe_state_dict
                except Exception as e:
                    print(f"Warning: Could not save model state_dict: {e}")
                    print("Saving model separately...")
                    # Save model separately if state_dict has issues
                    model_path = filename.replace('.pth.tar', '_model_only.pth')
                    torch.save(value, model_path)
                    safe_state[key] = f"Model saved separately to: {model_path}"
            
            elif key == 'optimizer':
                # Handle optimizer state_dict
                try:
                    if hasattr(value, 'state_dict'):
                        safe_state[key] = value.state_dict()
                    else:
                        safe_state[key] = value
                except Exception as e:
                    print(f"Warning: Could not save optimizer state: {e}")
                    safe_state[key] = "Optimizer state could not be saved"
            
            else:
                # Copy other items directly
                try:
                    # Try to copy the item
                    safe_state[key] = copy.deepcopy(value)
                except Exception:
                    # If deepcopy fails, try regular copy
                    try:
                        safe_state[key] = copy.copy(value)
                    except Exception:
                        # If all else fails, convert to basic type
                        if isinstance(value, (int, float, str, bool)):
                            safe_state[key] = value
                        else:
                            safe_state[key] = str(value)
        
        # Save the safe state
        torch.save(safe_state, filename)
        print(f"Checkpoint saved successfully to {filename}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        print("Attempting alternative save method...")
        
        # Alternative: Save only essential components
        try:
            minimal_state = {
                'epoch': state.get('epoch', 0),
                'best_dice': state.get('best_dice', 0.0),
                'train_losses': state.get('train_losses', []),
                'val_dices': state.get('val_dices', [])
            }
            
            # Try to save model state_dict separately
            if 'state_dict' in state:
                model_path = filename.replace('.pth.tar', '_model.pth')
                torch.save(state['state_dict'], model_path)
                minimal_state['model_path'] = model_path
                print(f"Model state_dict saved to: {model_path}")
            
            # Try to save optimizer separately
            if 'optimizer' in state:
                try:
                    optimizer_path = filename.replace('.pth.tar', '_optimizer.pth')
                    if hasattr(state['optimizer'], 'state_dict'):
                        torch.save(state['optimizer'].state_dict(), optimizer_path)
                    else:
                        torch.save(state['optimizer'], optimizer_path)
                    minimal_state['optimizer_path'] = optimizer_path
                    print(f"Optimizer state saved to: {optimizer_path}")
                except Exception as opt_e:
                    print(f"Could not save optimizer: {opt_e}")
            
            # Save minimal state
            torch.save(minimal_state, filename)
            print(f"Minimal checkpoint saved to {filename}")
            
        except Exception as final_e:
            print(f"All save attempts failed: {final_e}")
            print("Training will continue but checkpoints cannot be saved.")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Safe checkpoint loading that handles different save formats
    """
    print("=> Loading checkpoint")
    
    try:
        if isinstance(checkpoint_path, str):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = checkpoint_path
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            if isinstance(checkpoint['state_dict'], str):
                # Model was saved separately
                print(f"Loading model from separate file: {checkpoint['state_dict']}")
                model_state = torch.load(checkpoint['state_dict'].split(': ')[-1], map_location='cpu')
                model.load_state_dict(model_state)
            else:
                # Normal state_dict
                model.load_state_dict(checkpoint['state_dict'])
        
        elif 'model_path' in checkpoint:
            # Model saved separately in minimal format
            print(f"Loading model from: {checkpoint['model_path']}")
            model_state = torch.load(checkpoint['model_path'], map_location='cpu')
            model.load_state_dict(model_state)
        
        # Load optimizer if provided
        if optimizer and 'optimizer' in checkpoint:
            try:
                if isinstance(checkpoint['optimizer'], str):
                    if 'optimizer_path' in checkpoint:
                        print(f"Loading optimizer from: {checkpoint['optimizer_path']}")
                        opt_state = torch.load(checkpoint['optimizer_path'], map_location='cpu')
                        optimizer.load_state_dict(opt_state)
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
        
        # Print loading info
        if 'epoch' in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'best_dice' in checkpoint:
            print(f"Best dice score was: {checkpoint['best_dice']:.4f}")
            
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
    ):
    
    train_ds = RoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=safe_collate_fn  # Custom collate function eklendi
    )

    val_ds = RoadDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=safe_collate_fn  # Custom collate function eklendi
    )

    return train_loader, val_loader

def calculate_metrics(preds, targets, num_classes=2):
    """Detailed metrics calculation"""
    metrics = {}
    
    # Overall accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()
    metrics['accuracy'] = correct / total
    
    # Per-class metrics
    class_metrics = {}
    
    for class_idx in range(num_classes):
        pred_class = (preds == class_idx)
        true_class = (targets == class_idx)
        
        # True Positives, False Positives, False Negatives
        # ~ is used for logical NOT operation
        tp = (pred_class & true_class).sum().item()
        fp = (pred_class & ~true_class).sum().item()
        fn = (~pred_class & true_class).sum().item()
        tn = (~pred_class & ~true_class).sum().item()
        
        # Precision, Recall, F1, IoU, Dice
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8) # f1 and dice score give same result for binary classification
        iou = tp / (tp + fp + fn + 1e-8) 
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8) # f1 and dice score give same result for binary classification
        
        class_metrics[f'class_{class_idx}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'dice': dice,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    metrics['per_class'] = class_metrics
    
    # Mean metrics
    mean_precision = np.mean([class_metrics[f'class_{i}']['precision'] for i in range(num_classes)])
    mean_recall = np.mean([class_metrics[f'class_{i}']['recall'] for i in range(num_classes)])
    mean_f1 = np.mean([class_metrics[f'class_{i}']['f1'] for i in range(num_classes)])
    mean_iou = np.mean([class_metrics[f'class_{i}']['iou'] for i in range(num_classes)])
    mean_dice = np.mean([class_metrics[f'class_{i}']['dice'] for i in range(num_classes)])
    
    metrics['mean_precision'] = mean_precision
    metrics['mean_recall'] = mean_recall
    metrics['mean_f1'] = mean_f1
    metrics['mean_iou'] = mean_iou
    metrics['mean_dice'] = mean_dice
    
    return metrics

def check_accuracy(loader, model, device="cuda", num_classes=2, save_confusion_matrix=False):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            # Predictions
            outputs = model(x)
            preds = torch.softmax(outputs, dim=1)
            preds = torch.argmax(preds, dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate detailed metrics
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    
    # Print results
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Mean Dice Score: {metrics['mean_dice']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")
    
    print("\nPer-class metrics:")
    for i in range(num_classes):
        class_metrics = metrics['per_class'][f'class_{i}']
        print(f"Class {i}: Dice={class_metrics['dice']:.4f}, IoU={class_metrics['iou']:.4f}, "
              f"Precision={class_metrics['precision']:.4f}, Recall={class_metrics['recall']:.4f}")
    
    # Save confusion matrix
    if save_confusion_matrix:
        plot_confusion_matrix(all_targets.numpy(), all_preds.numpy(), num_classes)
    
    model.train()
    return metrics['mean_dice']

def plot_confusion_matrix(y_true, y_pred, num_classes=4, save_path="confusion_matrix.png"):
    """Confusion matrix çizimi"""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_predictions_as_imgs(loader, model, folder="saved_images_new/", device="cuda", num_samples=5):
    """Geliştirilmiş prediction kaydetme"""
    model.eval()
    os.makedirs(folder, exist_ok=True)
    
    # Class colors for visualization

    colors = np.array([
        [0, 0, 0],      # Class 0 - Black
        [0, 255, 0],    # Class 1 - Green
    ])
    
    sample_count = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        if sample_count >= num_samples:
            break
            
        x = x.to(device=device)
        
        with torch.no_grad():
            outputs = model(x)
            preds = torch.softmax(outputs, dim=1)
            preds = torch.argmax(preds, dim=1)

        # Save each image in batch
        batch_size = x.shape[0]
        for i in range(min(batch_size, num_samples - sample_count)):
            # Original image (denormalize)
            img = x[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            # Prediction and ground truth
            pred = preds[i].cpu().numpy()
            target = y[i].cpu().numpy()
            
            # Create colored masks
            pred_colored = colors[pred]
            target_colored = colors[target]
            
            # Save images
            torchvision.utils.save_image(img, f"{folder}/original_{sample_count}.png")
            
            # Convert to tensor format for saving
            pred_tensor = torch.from_numpy(pred_colored).permute(2, 0, 1).float() / 255.0
            target_tensor = torch.from_numpy(target_colored).permute(2, 0, 1).float() / 255.0
            
            torchvision.utils.save_image(pred_tensor, f"{folder}/pred_{sample_count}.png")
            torchvision.utils.save_image(target_tensor, f"{folder}/target_{sample_count}.png")
            
            sample_count += 1
            
        if sample_count >= num_samples:
            break
    
    print(f"Saved {sample_count} prediction samples to {folder}")
    model.train()

def test_time_augmentation(model, x, device="cuda"):
    """Test Time Augmentation"""
    model.eval()
    
    with torch.no_grad():
        # Original
        pred1 = torch.softmax(model(x), dim=1)
        
        # Horizontal flip
        x_hflip = torch.flip(x, dims=[3])
        pred2 = torch.softmax(model(x_hflip), dim=1)
        pred2 = torch.flip(pred2, dims=[3])
        
        # Vertical flip  
        x_vflip = torch.flip(x, dims=[2])
        pred3 = torch.softmax(model(x_vflip), dim=1)
        pred3 = torch.flip(pred3, dims=[2])
        
        # Average predictions
        pred_avg = (pred1 + pred2 + pred3) / 3.0
        
    return pred_avg

def create_training_plots(train_losses, val_dices, save_path="training_plots.png"):
    """Visualize training losses and validation dice scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Dice score plot
    ax2.plot(val_dices, label='Validation Dice Score', color='red')
    ax2.set_title('Validation Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {save_path}")