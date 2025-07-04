import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from improved_unet import ImprovedUNET

def load_model(checkpoint_path, device):
    """Load model and weights"""
    # Initialize the model (adjust in_channels and out_channels as per your model)
    model = ImprovedUNET(in_channels=3, out_channels=2)
    
    # Checkpoint loading
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load the state dict 
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Show checkpoint information (for debugging)
    if 'epoch' in checkpoint:
        print(f"Loaded model - Epoch: {checkpoint['epoch']}")
    if 'best_dice' in checkpoint:
        print(f"Best Dice Score: {checkpoint['best_dice']:.4f}")
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, input_size=(480, 320)):
    """preprocess the input image for the model"""
    # Load the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image couldn't be loaded: {image_path}")
        # Converting BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if len(image_path.shape) == 3 and image_path.shape[2] == 3:
            # Converting BGR to RGB
            image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_path
        image = image_path
    
    pil_image = Image.fromarray(image_rgb)
    
    # Transform operations
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to tensor and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)
    return tensor, image

def postprocess_mask(mask, original_shape):
    """Process the mask and visualize it"""
    # Tensor'dan numpy'a çevir
    mask_np = mask.squeeze().cpu().numpy()  # Shape: (2, H, W)
    
    # Implement softmax
    mask_np = np.exp(mask_np) / np.sum(np.exp(mask_np), axis=0, keepdims=True)
    
    # Get the highest probability class
    segmentation_map = np.argmax(mask_np, axis=0)  # Shape: (H, W)
    
    # Convert to 0-255 range and assign different colors for classes
    colored_mask = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    
    colors = [
        [0, 0, 0],       # Class 0: Black (Background)
        [0, 255, 0]      # Class 1: Green (Road)
    ]
    
    for i in range(2):
        mask_class = (segmentation_map == i)
        colored_mask[mask_class] = colors[i]
    
    # Resize to original shape
    colored_mask_resized = cv2.resize(colored_mask, (original_shape[1], original_shape[0]))
    
    return colored_mask_resized, segmentation_map

def process_single_image(image_path, model_path, save_result=False, output_path=None):
    """Inference on a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model yükle
    print("Model preparing...")
    model = load_model(model_path, device)
    
    # Görüntüyü hazırla
    print(f"Image processing: {image_path}")
    input_tensor, original_image = preprocess_image(image_path)
    
    # Inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        prediction = model(input_tensor)
        print(f"Model output shape: {prediction.shape}")
    
    # Process the mask
    colored_mask, segmentation_map = postprocess_mask(prediction, original_image.shape[:2])
    
    if len(original_image.shape) == 3:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original_image
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Visualizing results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_map, cmap='viridis')
    plt.title('Segmentation Map')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_rgb)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save results if required
    if save_result:
        if output_path is None:
            # Automatically generate output path if not provided
            base_name = image_path.split('/')[-1].split('.')[0] if isinstance(image_path, str) else "test_image"
            output_path = f"{base_name}_segmentation_result.jpg"
        
        # Save the overlay
        cv2.imwrite(output_path, overlay)
        print(f"Vverlay Saved: {output_path}")
        
        # Save the segmentation map 
        seg_output_path = output_path.replace('.jpg', '_segmap.jpg')
        plt.imsave(seg_output_path, segmentation_map, cmap='viridis')
        print(f"Segmentation Map Saved: {seg_output_path}")
    
    # Show segmentation statistics
    unique_classes, counts = np.unique(segmentation_map, return_counts=True)
    total_pixels = segmentation_map.shape[0] * segmentation_map.shape[1]
    
    print("\n--- Segmentation Statistics ---")
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = "Background" if class_id == 0 else f"Sınıf {class_id}"
        print(f"{class_name}: {count} pixels ({percentage:.2f}%)")
    
    return overlay_rgb, segmentation_map, colored_mask

def test_multiple_images(image_paths, model_path):
    """Test multiple images with the segmentation model."""
    for i, image_path in enumerate(image_paths):
        print(f"\n{'='*50}")
        print(f"Test {i+1}/{len(image_paths)}: {image_path}")
        print(f"{'='*50}")
        
        try:
            overlay, seg_map, mask = process_single_image(
                image_path, 
                model_path, 
                save_result=False,
                output_path=f"test_result_{i+1}.jpg"
            )
            print(f"✓ Test {i+1} successed!")
        except Exception as e:
            print(f"✗ Test {i+1} failed: {str(e)}")


if __name__ == "__main__":
    # Model path
    model_path = "checkpoints/best_model.pth.tar"
    test_images = [
        "challanges_image/test_image1.jpg",
        "challanges_image/test_image2.jpg", 
        "challanges_image/test_image3.jpg",
        "challanges_image/test_image4.jpg",
        "challanges_image/test_image5.jpg",
        "challanges_image/test_image6.jpg"
    ]
    
    print("\nTestring multiple images with the segmentation model...")
    test_multiple_images(test_images, model_path)


    
    # Tek görüntü testi
    """
    image_path = "challanges_image/test_image5.jpg"  
    
    try:
        overlay, segmentation_map, colored_mask = process_single_image(
            image_path=image_path,
            model_path=model_path,
            save_result=False,
            output_path="single_test_result.jpg"
        )
        print("✓ Test successed!")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
    """
    
    
    