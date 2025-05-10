import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        
        # Classification head
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input features tensor of shape (C, H, W)
        Returns:
            x: Output tensor of shape (H*W, 1) for binary classification
        """
        # Input is already in (C, H, W) format from dataset
        # Add batch dimension
        x = x.unsqueeze(0)  # (1, C, H, W)
        
        # Store input size for later
        input_size = x.shape[2:]
        
        # Encoder path with size tracking
        x1 = self.inc(x)                                     # (1, 64, H, W)
        x2 = self.down1(x1)                                  # (1, 128, H/2, W/2)
        x3 = self.down2(x2)                                  # (1, 256, H/4, W/4)
        x4 = self.down3(x3)                                  # (1, 512, H/8, W/8)
        
        # Decoder path with careful size matching
        x = self.up1(x4)                                     # (1, 256, H/4, W/4)
        x = F.interpolate(x, size=x3.shape[2:])             # Ensure exact size match
        x = torch.cat([x, x3], dim=1)                       # (1, 512, H/4, W/4)
        x = self.up_conv1(x)                                # (1, 256, H/4, W/4)
        
        x = self.up2(x)                                     # (1, 128, H/2, W/2)
        x = F.interpolate(x, size=x2.shape[2:])             # Ensure exact size match
        x = torch.cat([x, x2], dim=1)                       # (1, 256, H/2, W/2)
        x = self.up_conv2(x)                                # (1, 128, H/2, W/2)
        
        x = self.up3(x)                                     # (1, 64, H, W)
        x = F.interpolate(x, size=x1.shape[2:])             # Ensure exact size match
        x = torch.cat([x, x1], dim=1)                       # (1, 128, H, W)
        x = self.up_conv3(x)                                # (1, 64, H, W)
        
        # Final convolution
        x = self.outc(x)                                    # (1, 1, H, W)
        
        # Ensure output size matches input size exactly
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # Reshape to (H*W, 1) for binary classification
        x = x.view(-1, self.n_classes)                      # (H*W, 1)
        
        return x
