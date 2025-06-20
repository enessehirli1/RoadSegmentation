import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from improved_unet import ImprovedUNET

def load_model(checkpoint_path, device):
    """Load model and weights from checkpoint"""
    # Ini,tialize the model
    model = ImprovedUNET(in_channels=3, out_channels=2)
    
    # Checkpoint loading
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # State dict loading
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
        print(f"Best Dice score: {checkpoint['best_dice']:.4f}")
    
    model.to(device)
    model.eval()
    return model

def preprocess_frame(frame, input_size=(480, 320)):
    # BGR'dan RGB'ye çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # PIL Image'a çevir
    pil_image = Image.fromarray(frame_rgb)
    
    # Transform işlemleri
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Tensor'a çevir ve batch dimension ekle
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def postprocess_mask(mask, original_shape):
    # Convert mask tensor to numpy array
    mask_np = mask.squeeze().cpu().numpy()  
    
    # Softmax 
    mask_np = np.exp(mask_np) / np.sum(np.exp(mask_np), axis=0, keepdims=True)
    
    # Get the highest probability class for each pixel
    segmentation_map = np.argmax(mask_np, axis=0)  # Shape: (H, W)
    
    # Convert to 0-255 range and different colors for different classes
    colored_mask = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    
    # Her sınıf için farklı renk
    colors = [
        [0, 0, 0],       # Sınıf 0: Siyah (background)
        [0, 255, 0]      # Sınıf 2: Yeşil
    ]
    
    for i in range(2):
        mask_class = (segmentation_map == i)
        colored_mask[mask_class] = colors[i]
    
    # Resize to original frame size
    colored_mask_resized = cv2.resize(colored_mask, (original_shape[1], original_shape[0]))
    
    return colored_mask_resized

def process_video(video_path, model_path, output_path=None, show_live=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model yükle
    print("Model loading...")
    model = load_model(model_path, device)
    
    # Video aç
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Video couldn't be opened!")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Informations: {width}x{height}, {fps} FPS, {total_frames} frame")
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    # Set matplotlib figure (for live display)
    if show_live:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.set_title('Original Frame')
        ax2.set_title('Segmentation Result')
        ax1.axis('off')
        ax2.axis('off')
    
    frame_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processed frame: {frame_count}/{total_frames}", end='\r')
            
            # Prepeare the frame
            input_tensor = preprocess_frame(frame).to(device)
            
            # Inference
            prediction = model(input_tensor)
            
            # Process the mask
            colored_mask = postprocess_mask(prediction, frame.shape[:2])
            
            # Merge the mask with the original frame
            overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
            
            # Display side by side
            combined = np.hstack([frame, overlay])
            
            # Show with matplotlib (live)
            if show_live:
                # BGR to RGB conversion for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                ax1.clear()
                ax2.clear()
                ax1.imshow(frame_rgb)
                ax2.imshow(overlay_rgb)
                ax1.set_title('Original Frame')
                ax2.set_title('Segmentation Result')
                ax1.axis('off')
                ax2.axis('off')
                
                plt.pause(0.01)  
            
            # Write to output video (if specified)
            if output_path:
                out.write(combined)
            
            # Her 10 frame'de bir kullanıcıya devam etmek isteyip istemediğini sor
            if show_live and frame_count % 10 == 0:
                plt.draw()
            
    # Kaynakları temizle
    cap.release()
    if output_path:
        out.release()
    
    if show_live:
        plt.ioff()
        plt.show()
    
    print(f"Total {frame_count} frame processed.")

# Ana fonksiyon
if __name__ == "__main__":
    video_path = "challanges_video/t_b.mp4"
    model_path = "checkpoints/best_model.pth.tar"
    output_path = "output_segmentation.mp4"  # output video path

    process_video(video_path, model_path, output_path, show_live=True)