"""
FaceForge AI - Deep Convolutional Generative Adversarial Network (DCGAN)
Implements Generator and Discriminator neural networks following Radford et al. (2015)
optimized for synthetic facial synthesis on CelebA.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

def weights_init(m):
    """
    Custom weights initialization according to DCGAN paper:
    All weights are randomly initialized from N(0, 0.02),
    BatchNorm weights initialized to N(1, 0.02) and bias to 0.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    DCGAN Generator Network G(z):
    Transforms a 100-dimensional random latent vector z into a 64x64x3 synthetic human face image.
    Architecture:
      Input: z in R^{100x1x1}
      Layer 1: ConvTranspose2d -> (ngf*8) x 4 x 4 (512 x 4 x 4) + BN + ReLU
      Layer 2: ConvTranspose2d -> (ngf*4) x 8 x 8 (256 x 8 x 8) + BN + ReLU
      Layer 3: ConvTranspose2d -> (ngf*2) x 16 x 16 (128 x 16 x 16) + BN + ReLU
      Layer 4: ConvTranspose2d -> (ngf) x 32 x 32 (64 x 32 x 32) + BN + ReLU
      Layer 5: ConvTranspose2d -> 3 x 64 x 64 + Tanh
    """
    def __init__(self, latent_dim: int = 100, ngf: int = 64, nc: int = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1, going into first 4x4 spatial projection
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output state size: (nc) x 64 x 64 with values normalized in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.main(x)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network D(x):
    Evaluates whether an input 64x64x3 face image is real (from training data) or fake (synthesized by G).
    Architecture:
      Input: x in R^{3x64x64}
      Layer 1: Conv2d -> (ndf) x 32 x 32 + LeakyReLU(0.2) (No BatchNorm on first layer)
      Layer 2: Conv2d -> (ndf*2) x 16 x 16 + BN + LeakyReLU(0.2)
      Layer 3: Conv2d -> (ndf*4) x 8 x 8 + BN + LeakyReLU(0.2)
      Layer 4: Conv2d -> (ndf*8) x 4 x 4 + BN + LeakyReLU(0.2)
      Layer 5: Conv2d -> 1 x 1 x 1 + Sigmoid
    """
    def __init__(self, nc: int = 3, ndf: int = 64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: Scalar probability in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1)


def get_model_summary(model: nn.Module, model_type: str = "generator") -> Dict[str, Any]:
    """Inspect model layers, shapes, and parameter counts for Model Explorer."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layers_info = []
    layer_idx = 1
    
    if model_type == "generator":
        layers_info = [
            {"index": 1, "name": "Latent Input (z)", "type": "Input Tensor", "output_shape": "[B, 100, 1, 1]", "params": 0, "activation": "None", "notes": "Gaussian latent vector z ~ N(0, I)"},
            {"index": 2, "name": "ConvTranspose2d_1", "type": "Fractionally-strided Conv", "output_shape": "[B, 512, 4, 4]", "params": 819200, "activation": "ReLU + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 1, Pad: 0"},
            {"index": 3, "name": "ConvTranspose2d_2", "type": "Fractionally-strided Conv", "output_shape": "[B, 256, 8, 8]", "params": 2097152, "activation": "ReLU + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 2, Pad: 1"},
            {"index": 4, "name": "ConvTranspose2d_3", "type": "Fractionally-strided Conv", "output_shape": "[B, 128, 16, 16]", "params": 524288, "activation": "ReLU + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 2, Pad: 1"},
            {"index": 5, "name": "ConvTranspose2d_4", "type": "Fractionally-strided Conv", "output_shape": "[B, 64, 32, 32]", "params": 131072, "activation": "ReLU + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 2, Pad: 1"},
            {"index": 6, "name": "ConvTranspose2d_5 (Output)", "type": "Fractionally-strided Conv", "output_shape": "[B, 3, 64, 64]", "params": 3072, "activation": "Tanh", "notes": "Channels: 3 (RGB), Output range: [-1, 1]"}
        ]
    else:
        layers_info = [
            {"index": 1, "name": "Input Image (x)", "type": "RGB Image Tensor", "output_shape": "[B, 3, 64, 64]", "params": 0, "activation": "None", "notes": "Synthetic or Real Face"},
            {"index": 2, "name": "Conv2d_1", "type": "Convolutional Layer", "output_shape": "[B, 64, 32, 32]", "params": 3072, "activation": "LeakyReLU(0.2)", "notes": "Kernel: 4x4, Stride: 2, Pad: 1 (No BN)"},
            {"index": 3, "name": "Conv2d_2", "type": "Convolutional Layer", "output_shape": "[B, 128, 16, 16]", "params": 131072, "activation": "LeakyReLU(0.2) + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 2, Pad: 1"},
            {"index": 4, "name": "Conv2d_3", "type": "Convolutional Layer", "output_shape": "[B, 256, 8, 8]", "params": 524288, "activation": "LeakyReLU(0.2) + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 2, Pad: 1"},
            {"index": 5, "name": "Conv2d_4", "type": "Convolutional Layer", "output_shape": "[B, 512, 4, 4]", "params": 2097152, "activation": "LeakyReLU(0.2) + BatchNorm2d", "notes": "Kernel: 4x4, Stride: 2, Pad: 1"},
            {"index": 6, "name": "Conv2d_5 (Classification)", "type": "Convolutional Layer", "output_shape": "[B, 1, 1, 1]", "params": 8192, "activation": "Sigmoid", "notes": "Binary output probability D(x) in [0, 1]"}
        ]
        
    return {
        "model_type": model_type,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layers": layers_info
    }
