import torch
import torch.nn as nn
import torch.nn.functional as F
from .mimevit_parts import *

class MIMEViT(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super(MIMEViT, self).__init__()
        self.n_classes = n_classes
        self.inc = (DoubleConv(1, 16))
        # self.vit_512 = ViT(1, 512, 192, 6, 6, 128, 1024, dropout=0.1)
        self.vit_256 = ViT(1, 512, 192, 6, 6, 128, scale_factor=2, dropout=0.1)
        self.vit_64 = ViT(1, 512, 384, 8, 8, 256, scale_factor=2, dropout=0.1)
        self.vit_16 = ViT(1, 512, 192, 8, 6, 64, scale_factor=2, dropout=0.1)
        
        self.residual_block = ResidualBlock(1, 1, stride=3)
        
        factor = 2 if bilinear else 1
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x): # x.shape: [B, 1, 2048, 2048]
        x1 = self.inc(x)                 # torch.Size([B, 64, 2048, 2048])
        # x_512 = self.vit_512(x)          # torch.Size([B, 16, 192])
        x_256 = self.vit_256(x)          # torch.Size([B, 64, 192]) 
        x_64 = self.vit_64(x)            # torch.Size([B, 1024, 384])
        x_16 = self.vit_16(x)            # torch.Size([B, 961, 96])
        
        x_conv = self.residual_block(x)  # torch.Size([B, 192, 16])
                
        # Now pass the input through the UNet blocks
        x = self.up1(x_64, x_256)  # Up-sample and concatenate
        x = self.up2(x, x_16)  # Up-sample and concatenate
        x = self.up3(x, x_conv)  # Up-sample and concatenate
        x = self.up4(x, x1)  # Up-sample and concatenate
        x = self.outc(x)  # Get the final output
        return x