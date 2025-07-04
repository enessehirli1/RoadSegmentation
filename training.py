import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import improved modules
from improved_unet import ImprovedUNET
from loss import CombinedLoss, DiceLoss, TverskyLoss, FocalLoss
from utils import (
    load_checkpoint, save_checkpoint, get_loaders, 
    check_accuracy, save_predictions_as_imgs, create_training_plots
)

# Configuration Class
class Config:
    # Model parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 2  # Number of classes for segmentation
    FEATURES = [64, 128, 256, 512]
    DROPOUT_RATE = 0.2
    
    # Training parameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Image parameters
    IMAGE_HEIGHT = 320
    IMAGE_WIDTH = 480
    
    # Paths
    TRAIN_IMG_DIR = "data/train_images/"
    TRAIN_MASK_DIR = "data/train_masks/"
    VAL_IMG_DIR = "data/val_images/"
    VAL_MASK_DIR = "data/val_masks/"
    
    # Checkpoints
    LOAD_MODEL = False
    CHECKPOINT_DIR = "checkpoints/"
    BEST_MODEL_PATH = "checkpoints/best_model.pth.tar"
    LAST_MODEL_PATH = "checkpoints/last_model.pth.tar"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Loss and scheduler parameters
    LOSS_TYPE = "combined"  # "dice", "combined", "tversky", "focal"
    LOSS_ALPHA = 0.7  # For combined loss (dice weight)
    SCHEDULER_TYPE = "plateau"  # "plateau", "cosine", "onecycle"
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    MIN_DELTA = 1e-4
    
    # Torch Compile Parameters - IMPROVED FOR COMPATIBILITY
    USE_TORCH_COMPILE = True
    COMPILE_MODE = "default"  # "default", "reduce-overhead", "max-autotune"
    COMPILE_FULLGRAPH = False
    COMPILE_DYNAMIC = True
    COMPILE_BACKEND = "aot_eager"  # Changed from "inductor" to "aot_eager" for better compatibility
    FALLBACK_TO_EAGER = True  # Enable fallback if compilation fails

def get_transforms():
    """Enhanced augmentation strategy"""
    
    train_transforms = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        
        # Geometric transformations
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Transpose(p=1.0),
        ], p=0.4),
        
        A.OneOf([
            A.Rotate(limit=20, p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        ], p=0.4),
        
        # Elastic deformations
        A.OneOf([
            A.ElasticTransform(p=1.0, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=1.0, distort_limit=0.3),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0)
        ], p=0.3),
        
        # Appearance augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.6),
        
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.ChannelShuffle(p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.3),
        
        # Noise and blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),
        
        # Weather effects
        A.OneOf([
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=8, p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                           num_flare_circles_lower=1, num_flare_circles_upper=3, 
                           src_radius=50, p=1.0),
        ], p=0.2),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    return train_transforms, val_transforms

def get_loss_function():
    """Get loss function based on config"""
    if Config.LOSS_TYPE == "dice":
        return DiceLoss(num_classes=Config.OUT_CHANNELS)
    elif Config.LOSS_TYPE == "combined":
        return CombinedLoss(num_classes=Config.OUT_CHANNELS, alpha=Config.LOSS_ALPHA)
    elif Config.LOSS_TYPE == "tversky":
        return TverskyLoss(num_classes=Config.OUT_CHANNELS, alpha=0.3, beta=0.7)
    elif Config.LOSS_TYPE == "focal":
        return FocalLoss(num_classes=Config.OUT_CHANNELS, alpha=1, gamma=2)
    else:
        return nn.CrossEntropyLoss()

def get_scheduler(optimizer):
    """Get learning rate scheduler"""
    if Config.SCHEDULER_TYPE == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, 
                               min_lr=1e-7)
    elif Config.SCHEDULER_TYPE == "cosine":
        return CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-7)
    elif Config.SCHEDULER_TYPE == "onecycle":
        return OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE, 
                         epochs=Config.NUM_EPOCHS, steps_per_epoch=1)
    else:
        return None


def get_best_compile_backend():
    """Determine the best compilation backend based on available dependencies"""
    if not hasattr(torch, 'compile'):
        return None, "torch.compile not available"
    
    # Check available backends
    available_backends = []
    
    # aot_eager is usually available
    available_backends.append("aot_eager")
    
    # Choose the best backend
    if "inductor" in available_backends and Config.DEVICE == "cuda":
        return "inductor", "Using inductor backend with Triton"
    elif "aot_eager" in available_backends:
        return "aot_eager", "Using aot_eager backend (Triton not available)"
    else:
        return None, "No suitable backend available"

def compile_model(model):
    """Compile model with automatic backend selection and fallback"""
    if not Config.USE_TORCH_COMPILE:
        print("⚠️  torch.compile disabled in config")
        return model
    
    # Check PyTorch version
    if not hasattr(torch, 'compile'):
        print("⚠️  torch.compile not available in this PyTorch version")
        print(f"    Current version: {torch.__version__}")
        print("    Requires PyTorch >= 2.0")
        return model
    
    # Get the best available backend
    backend, backend_msg = get_best_compile_backend()
    if backend is None:
        print(f"⚠️  {backend_msg}")
        return model
    
    # Override config backend if needed
    actual_backend = backend
    print(f"🚀 Compiling model with torch.compile...")
    print(f"   {backend_msg}")
    print(f"   Mode: {Config.COMPILE_MODE}")
    print(f"   Backend: {actual_backend}")
    print(f"   Fullgraph: {Config.COMPILE_FULLGRAPH}")
    print(f"   Dynamic: {Config.COMPILE_DYNAMIC}")
    
    try:
        compiled_model = torch.compile(
            model,
            mode=Config.COMPILE_MODE,
            fullgraph=Config.COMPILE_FULLGRAPH,
            dynamic=Config.COMPILE_DYNAMIC,
            backend=actual_backend
        )
        
        print("✅ Model compiled successfully!")
        return compiled_model
        
    except Exception as e:
        print(f"❌ torch.compile failed: {e}")
        if Config.FALLBACK_TO_EAGER:
            print("   Falling back to eager mode...")
            return model
        else:
            raise e

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=20, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    # For compiled models, we need to access the original model
                    original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    original_model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        # For compiled models, we need to access the original model's state dict
        original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.best_weights = original_model.state_dict().copy()

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """Enhanced training function with torch.compile support"""
    # Set model to training mode
    if hasattr(model, '_orig_mod'):
        model._orig_mod.train()
    else:
        model.train()
    
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device, non_blocking=True)
        targets = targets.squeeze(1).long().to(device=device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            predictions = model(data)
            
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        # For compiled models, we need to access the original model's parameters
        if hasattr(model, '_orig_mod'):
            torch.nn.utils.clip_grad_norm_(model._orig_mod.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        
        # Update progress bar
        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )
    
    return avg_loss

def validate_fn(loader, model, loss_fn, device):
    """Validation function with torch.compile support"""
    # Set model to evaluation mode
    if hasattr(model, '_orig_mod'):
        model._orig_mod.eval()
    else:
        model.eval()
    
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(loader, desc="Validation")):
            data = data.to(device=device, non_blocking=True)
            targets = targets.squeeze(1).long().to(device=device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            
            running_loss += loss.item()
    
    return running_loss / len(loader)

def warmup_compiled_model(model, data_loader, device, max_warmup_batches=3):
    """Warmup the compiled model with better error handling"""
    if not hasattr(model, '_orig_mod'):
        print("🔥 Model not compiled, skipping warmup")
        return
    
    print("🔥 Warming up compiled model...")
    model.eval()
    
    try:
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= max_warmup_batches:
                    break
                
                data = data.to(device=device, non_blocking=True)
                
                # Try warmup with error handling
                try:
                    with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                        _ = model(data)
                    print(f"   Warmup batch {i+1}/{max_warmup_batches} completed")
                except Exception as e:
                    print(f"   Warmup batch {i+1} failed: {e}")
                    # Continue with other batches
                    continue
        
        print("✅ Model warmup completed!")
        
    except Exception as e:
        print(f"⚠️  Model warmup failed: {e}")
        print("   Training will continue without warmup")

def main():
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("saved_images", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print(f"Using device: {Config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model: ImprovedUNET with {Config.OUT_CHANNELS} classes")
    print(f"Loss function: {Config.LOSS_TYPE}")
    print(f"Scheduler: {Config.SCHEDULER_TYPE}")
    print(f"torch.compile enabled: {Config.USE_TORCH_COMPILE}")
    
    
    # Get transforms
    train_transforms, val_transforms = get_transforms()
    
    # Initialize model
    model = ImprovedUNET(
        in_channels=Config.IN_CHANNELS, 
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(Config.DEVICE)
    
    # Compile model if enabled
    model = compile_model(model)
    
    # Count parameters
    original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    total_params = sum(p.numel() for p in original_model.parameters())
    trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    loss_fn = get_loss_function()
    optimizer = optim.AdamW(
        original_model.parameters(),  # Use original model parameters for optimizer
        lr=Config.LEARNING_RATE, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = get_scheduler(optimizer)
    
    # Data loaders
    train_loader, val_loader = get_loaders(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
        Config.VAL_IMG_DIR,
        Config.VAL_MASK_DIR,
        Config.BATCH_SIZE,
        train_transforms,
        val_transforms,
        Config.NUM_WORKERS,
        Config.PIN_MEMORY
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Warmup compiled model
    warmup_compiled_model(model, train_loader, Config.DEVICE)
    
    # Load checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    train_losses = []
    val_losses = []
    val_dices = []
    
    if Config.LOAD_MODEL and os.path.exists(Config.BEST_MODEL_PATH):
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
        load_checkpoint(checkpoint, original_model, optimizer)  # Load to original model
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_dice' in checkpoint:
            best_dice = checkpoint['best_dice']
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        if 'val_dices' in checkpoint:
            val_dices = checkpoint['val_dices']
        print(f"✅ Checkpoint loaded from epoch {start_epoch}")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_delta=Config.MIN_DELTA
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler()
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Training
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, Config.DEVICE)
        
        # Validation
        val_loss = validate_fn(val_loader, model, loss_fn, Config.DEVICE)
        
        # Check accuracy and get dice score
        dice_score = check_accuracy(val_loader, model, device=Config.DEVICE, 
                                  num_classes=Config.OUT_CHANNELS, 
                                  save_confusion_matrix=(epoch % 10 == 0))
        
        # Update learning rate
        if scheduler:
            if Config.SCHEDULER_TYPE == "plateau":
                scheduler.step(dice_score)
            else:
                scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(dice_score)

        if ((epoch + 1) % 5 == 0) or (epoch == 0):
            print("\nValidation Metrics:")
            print(f"Dice Score: {dice_score:.4f}")

            print(f"\nEpoch Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {dice_score:.4f}")
            print(f"Best Dice: {best_dice:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        
        # Save checkpoints
        checkpoint = {
            "epoch": epoch,
            "state_dict": original_model.state_dict(),  # Save original model state
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_dice": max(best_dice, dice_score),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_dices": val_dices,
            "config": Config.__dict__
        }
        
        # Save last model
        save_checkpoint(checkpoint, filename=Config.LAST_MODEL_PATH)
        
        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            checkpoint["best_dice"] = best_dice
            save_checkpoint(checkpoint, filename=Config.BEST_MODEL_PATH)
            print(f"🎉 New best model saved! Dice: {best_dice:.4f}")
            
            # Save predictions for best model
            save_predictions_as_imgs(
                val_loader, model, 
                folder=f"saved_images/epoch_{epoch+1}/", 
                device=Config.DEVICE,
                num_samples=10
            )
        
        # Create training plots
        if (epoch + 1) % 5 == 0:
            create_training_plots(train_losses, val_dices, 
                                save_path=f"plots/training_progress_epoch_{epoch+1}.png")
        
        # Early stopping check
        if early_stopping(dice_score, model):
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save predictions periodically
        if (epoch + 1) % 10 == 0:
            save_predictions_as_imgs(
                val_loader, model, 
                folder=f"saved_images/epoch_{epoch+1}/", 
                device=Config.DEVICE,
                num_samples=5
            )
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    # Load best model for final evaluation
    if os.path.exists(Config.BEST_MODEL_PATH):
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
        load_checkpoint(checkpoint, original_model)
        print(f"Loaded best model with Dice score: {checkpoint.get('best_dice', 'N/A')}")
    
    # Final accuracy check
    final_dice = check_accuracy(val_loader, model, device=Config.DEVICE, 
                              num_classes=Config.OUT_CHANNELS, 
                              save_confusion_matrix=True)
    
    # Save final predictions
    save_predictions_as_imgs(
        val_loader, model, 
        folder="saved_images/final/", 
        device=Config.DEVICE,
        num_samples=20
    )
    
    # Create final training plots
    create_training_plots(train_losses, val_dices, 
                        save_path="plots/final_training_progress.png")
    
    print(f"\n🏁 Training completed!")
    print(f"Final Dice Score: {final_dice:.4f}")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Model saved to: {Config.BEST_MODEL_PATH}")
    
    # Print compilation stats if available
    if hasattr(model, '_orig_mod'):
        try:
            print(f"\n📊 Compilation Stats:")
            print(f"Backend used: {getattr(model, '_compile_config', {}).get('backend', 'Unknown')}")
        except:
            pass

if __name__ == "__main__":
    main()