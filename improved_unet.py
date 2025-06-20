import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.final_conv(x)
        
        return x

class ImprovedUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[64, 128, 256, 512], dropout_rate=0.1):
        super(ImprovedUNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature

        # ASPP Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2, dropout_rate),
            ASPP(features[-1] * 2, features[-1])
        )

        # Attention blocks - Fix channel dimensions based on actual flow
        # features = [64, 128, 256, 512]
        # After upsampling: [512//2=256, 256//2=128, 128//2=64, 64//2=32]
        # Skip connections: [512, 256, 128, 64] (reversed from encoder)
        for i in range(len(features)):
            if i == 0:  # First attention block (from bottleneck)
                # g: upsampled from bottleneck (256), x: skip connection (512)
                self.attention_blocks.append(
                    AttentionBlock(features[-1]//2, features[-1], features[-1]//4)
                )
            else:
                # Subsequent attention blocks
                # g: upsampled features, x: skip connection from encoder
                g_channels = features[len(features)-1-i]//2  # upsampled channels
                x_channels = features[len(features)-1-i]     # skip connection channels
                self.attention_blocks.append(
                    AttentionBlock(g_channels, x_channels, min(g_channels, x_channels)//2)
                )

        # Up Part of UNET - Fix channel dimensions
        for i, feature in enumerate(reversed(features)):
            if i == 0:  # First upsampling (from bottleneck)
                # Bottleneck output: features[-1] -> upsample to feature//2
                self.ups.append(
                    nn.ConvTranspose2d(
                        features[-1], feature//2, kernel_size=2, stride=2
                    )
                )
                # Skip connection: feature + upsampled: feature//2 = feature + feature//2
                self.ups.append(DoubleConv(feature + feature//2, feature, dropout_rate))
            else:
                # Other upsampling layers
                prev_feature = features[len(features)-i]
                self.ups.append(
                    nn.ConvTranspose2d(
                        prev_feature, feature//2, kernel_size=2, stride=2
                    )
                )
                # Skip connection: feature + upsampled: feature//2 = feature + feature//2
                self.ups.append(DoubleConv(feature + feature//2, feature, dropout_rate))
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(features[0]//2, features[0]//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//4, out_channels, kernel_size=1)
        )

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder with attention
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Apply attention mechanism
            if idx // 2 < len(self.attention_blocks):
                skip_connection = self.attention_blocks[idx // 2](x, skip_connection)

            # Resize if necessary
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # Concatenate and apply double conv
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# Test function
def test():
    print("Testing ImprovedUNET...")
    x = torch.randn((2, 3, 320, 480))
    model = ImprovedUNET(in_channels=3, out_channels=4)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print("✅ Model test passed!")
    
    # Test with different input sizes
    test_sizes = [(1, 3, 256, 256), (1, 3, 512, 512), (4, 3, 224, 224)]
    for size in test_sizes:
        test_input = torch.randn(size)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Input {size} -> Output {test_output.shape}")

if __name__ == "__main__":
    test()