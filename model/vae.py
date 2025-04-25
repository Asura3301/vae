"""
This file contains the implementation of the VAE model.
Input img -> Encoder -> Latent space (mean, log_variance) -> Reparameterization trick -> z -> Decoder -> Output img
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    def __init__(self, n_head, n_embd, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(n_embd, 3 * n_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = n_embd // n_head

    def forward(self, x):
        # x: (Batch_Size, Seq_Len, Dim)
        batch_size, seq_len, d_embd = x.shape
        interim_shape = (batch_size, seq_len, self.n_head, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 * (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, N_Heads, Dim / N_Heads) -> (Batch_Size, N_Heads, Seq_Len, Dim / N_Heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # calculate the attention

        # attention (materializes the large(T, T) matrix for all the queries and keys)
        # in official gpt2 used `torch.baddbmm`- batch matrix-matrix product of matrice (it is a bit more efficient)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs) weighted sum of the tokens that model found interesting

        # More efficient implementation:
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # (Batch_Size, N_Heads, Seq_Len, Dim / N_Heads) -> (Batch_Size, Seq_Len, N_Heads, Dim / N_Heads)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, N_Heads, Dim / N_Heads) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape((batch_size, seq_len, d_embd))

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        return output

class AttentionBlock(nn.Module):
    
    """Attention block with Group Normalization and residual connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels) 

    def forward(self, x):
        # x: (batch_size, channels, h, w)
        residual = x.clone()
        N, C, H, W = x.shape

        x = self.groupnorm(x)

        # (N, C, H, W) -> (N, C, H * W)
        x = x.view((N, C, H * W))
        # (N, C, H * W) -> (N, H*W, C)
        x = x.transpose(-1, -2)

        # Perform self-attention without mask
        x = self.attention(x)

        # (N, H*W, C) -> (N, C, H*W)
        x = x.transpose(-1, -2)
        # (N, C, H*W) -> (N, C, H, W)
        x = x.view((N, C, H, W))

        x += residual
        return x

class ResidualBlock(nn.Module):
    
    """Residual block with convolutional layers and Group Normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # Ensure residual connection matches output channels
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        identity = x.clone() # residual path

        x = self.groupnorm1(x)
        x = F.silu(x) # Changed from F.selu to F.silu as commonly used()
        x = self.conv1(x)

        x = self.groupnorm2(x)
        x = F.silu(x) 
        x = self.conv2(x)

        x = x + self.residual_layer(identity)
        return x

class Encoder(nn.Sequential):
    """
    The Encoder layer of a variational autoencoder that encodes its
    input into a latent representation. Outputs mean and log_variance.
    """
    def __init__(self):
        super().__init__(
            # Input: (B, 3, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),            # -> (B, 128, H, W)
            ResidualBlock(128, 128),                                # -> (B, 128, H, W)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # -> (B, 128, H/2 -1, W/2 -1) Needs padding adjustment logic in forward
            ResidualBlock(128, 256),                                # -> (B, 256, H/2-1, W/2-1)
            ResidualBlock(256, 256),                                # -> (B, 256, H/2-1, W/2-1)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), # -> (B, 256, H/4-1, W/4-1) Needs padding adjustment logic
            ResidualBlock(256, 512),                                # -> (B, 512, H/4-1, W/4-1)
            ResidualBlock(512, 512),                                # -> (B, 512, H/4-1, W/4-1)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), # -> (B, 512, H/8-1, W/8-1) Needs padding adjustment logic
            ResidualBlock(512, 512),                                # -> (B, 512, H/8-1, W/8-1)
            ResidualBlock(512, 512),                                # -> (B, 512, H/8-1, W/8-1)
            ResidualBlock(512, 512),                                # -> (B, 512, H/8-1, W/8-1)
            AttentionBlock(512),                                    # -> (B, 512, H/8-1, W/8-1)
            ResidualBlock(512, 512),                                # -> (B, 512, H/8-1, W/8-1)
            nn.GroupNorm(32, 512),                                  # -> (B, 512, H/8-1, W/8-1)
            nn.SiLU(),                                              # -> (B, 512, H/8-1, W/8-1)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),            # -> (B, 8, H/8-1, W/8-1)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)               # -> (B, 8, H/8-1, W/8-1)
        )

    def forward(self, x): 
        # x: (B, 3, H, W)

        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                # Pad input before strided convolution to maintain size (H/2, W/2) etc.
                # Pad (left, right, top, bottom) - Pad by 1 on right/bottom for stride 2, kernel 3
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # x is now (B, 8, H/8, W/8) representing mean and log_variance
        # Split into mean and log_variance
        mean, log_variance = torch.chunk(x, 2, dim=1) # -> 2 * (B, 4, H/8, W/8)

        # Clamp log variance for stability between -30 to 20
        log_variance = torch.clamp(log_variance, -30, 20)

        std = torch.exp(0.5 * log_variance)

        # Reparameterization trick: z = mean + stdev * epsilon
        # Sample epsilon from standard normal distribution
        eps = torch.randn_like(std)
        x = mean + std * eps # Latent variable z

        # scaling latent representation constant
        x *= 0.18215
        
        return x



class Decoder(nn.Sequential):
    """
    The Decoder layer of a variational autoencoder that decodes its
    latent representation into an output sample.
    """
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),            # (batch_size, 4, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),                                # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            AttentionBlock(512),                                    # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),                                # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),                                # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),                                # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            nn.Upsample(scale_factor=2),                            # (batch_size, 512, 32, 32) -> (batch_size, 512, 64, 64)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),          # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            ResidualBlock(512, 512),                                # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            ResidualBlock(512, 512),                                # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            ResidualBlock(512, 512),                                # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            nn.Upsample(scale_factor=2),                            # (batch_size, 512, 64, 64) -> (batch_size, 512, 128, 128)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),          # (batch_size, 512, 128, 128) -> (batch_size, 512, 128, 128)
            ResidualBlock(512, 256),                                # (batch_size, 512, 128, 128) -> (batch_size, 256, 128, 128)
            ResidualBlock(256, 256),                                # (batch_size, 256, 128, 128) -> (batch_size, 256, 128, 128)
            ResidualBlock(256, 256),                                # (batch_size, 256, 128, 128) -> (batch_size, 256, 128, 128)
            nn.Upsample(scale_factor=2),                            # (batch_size, 256, 128, 128) -> (batch_size, 256, 256, 256)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),          # (batch_size, 512, 128, 128) -> (batch_size, 512, 128, 128)
            ResidualBlock(256, 128),                                # (batch_size, 256, 256, 256) -> (batch_size, 128, 256, 256)
            ResidualBlock(128, 128),                                # (batch_size, 128, 256, 256) -> (batch_size, 128, 256, 256)
            ResidualBlock(128, 128),                                # (batch_size, 128, 256, 256) -> (batch_size, 128, 256, 256)
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),            # (batch_size, 128, 256, 256) -> (batch_size, 3, 256, 256)
        )

    def forward(self, x):
        # x: (B, 4, H/8, W/8) - Latent variable z

        # Remove the scaling introduced by the Encoder
        x /= 0.18215 # Constant used in Stable Diffusion VAE

        for module in self:
            x = module(x)

        # Output: (B, 3, H, W) - Reconstructed image
        return x


class VAE(nn.Module):
    """Variational Autoencoder combining Encoder and Decoder."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # q_phi(z|x)
        encoded = self.encoder(x)
        # p_theta(x|z)
        decoded = self.decoder(encoded)
        
        return decoded, encoded
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    x = torch.randn(1, 3, 64, 64).to(device) # Ensure input tensor is on the same device
    decoded, encoded = vae(x) 

    print("Input shape:", x.shape)
    print("Decoded shape:", decoded.shape)
    print("Encoded shape:", encoded.shape) 

