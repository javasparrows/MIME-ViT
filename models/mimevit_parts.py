""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, embed_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, nhead = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  nhead
        project_out = not (nhead == 1 and dim_head == embed_dim)

        self.nhead = nhead
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.nhead), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_layers, nhead, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(num_layers):
            dim_head = embed_dim // nhead
            mlp_dim = embed_dim * 2  # or embed_dim * 3 if you prefer
            
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, Attention(embed_dim, nhead, dim_head, dropout)),
                PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout))
            ]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ResidualBlock(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, stride=1):
        """
        残差ブロックを作成するクラス
        Args:
            first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
            first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
            identity_conv : channel数調整用のconv層
            stride : 3×3conv層におけるstide数。sizeを半分にしたいときは2に設定
        """
        super(ResidualBlock, self).__init__()

        # 1番目のconv層（1×1）
        self.conv1 = nn.Conv2d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)

        # 2番目のconv層（3×3）
        # パターン3の時はsizeを変更できるようにstrideは可変
        self.conv2 = nn.Conv2d(
            first_conv_out_channels, 4, kernel_size=3, stride=2, padding=1)

        # 3番目のconv層（1×1）
        # output channelはinput channelの4倍になる
        self.conv3 = nn.Conv2d(
            4, 32, kernel_size=1, stride=2, padding=0)
        self.relu = nn.ReLU()

        # identityのchannel数の調整が必要な場合はconv層（1×1）を用意、不要な場合はNone
        # self.identity_conv = nn.Conv2d(1, 4, kernel_size=1, stride=3, padding=0)
        
        # self.ln = nn.LayerNorm(683)
        
        # self.conv_downsample = ConvDownsample()

    def forward(self, x):

        identity = x.clone()  # 入力を保持する

        x = self.conv1(x)  # 1×1の畳み込み
        x = self.relu(x)
        x = self.conv2(x)  # 3×3の畳み込み（パターン3の時はstrideが2になるため、ここでsizeが半分になる）
        x = self.relu(x)
        x = self.conv3(x)  # 1×1の畳み込み

        # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから足す
        # if self.identity_conv is not None:
        #     identity = self.identity_conv(identity)
        # x += identity
        
        # x = self.ln(x)
        # x = self.relu(x)
        # x = self.conv_downsample(x) # * self.x_conv_weight # Shape: [B, out_channels, 16]
        return x


class ConvDownsample(nn.Module):
    def __init__(self):
        super(ConvDownsample, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(8)
        self.conv2 = nn.Conv2d(32, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d((16, 1))
        # self.flatten = nn.Flatten(2, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # x = self.flatten(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, nhead, num_layers, output_channels, scale_factor=2, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.transformer = Transformer(embed_dim, num_layers, nhead, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(embed_dim, output_channels, kernel_size=1)  # 1x1 convolution
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # Upsampling to original resolution

    def forward(self, x):
        B, _, H, W = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        position_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim)).to(x.device)
        
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)  # Change shape to [B, H//patch_size, W//patch_size, embed_dim]
        
        x = self.dropout(x)
        x = self.transformer(x.view(B, -1, self.embed_dim))  # Flatten spatial dimensions
        x = self.norm(x)
        
        x = x.view(B, H//self.patch_size, W//self.patch_size, self.embed_dim)  # Reshape back to 2D
        x = x.permute(0, 3, 1, 2)  # Change shape to [B, embed_dim, H//patch_size, W//patch_size]
        
        x = self.conv_out(x)  # Apply 1x1 convolution to change channels to output_channels
        x = self.upsample(x)  # Upsample back to original resolution
        
        return x

class ViT16(ViT):
    def __init__(self, in_channels, patch_size, embed_dim, nhead, num_layers, factor, output_channels, dropout=0.1):
        super(ViT16, self).__init__(in_channels, patch_size, embed_dim, nhead, num_layers, factor, output_channels, dropout)
        self.conv = nn.Conv2d(1, 3, kernel_size=3, stride=2)
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = super().forward(x)
        return x
