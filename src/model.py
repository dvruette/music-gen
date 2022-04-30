import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

# https://github.com/milesial/Pytorch-UNet

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim=None, expansion=2):
        super().__init__()
        self.t_proj = nn.Linear(t_dim, in_channels) if t_dim is not None else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            LayerNorm(in_channels),
            nn.Conv2d(in_channels, expansion*out_channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(expansion*out_channels, out_channels, kernel_size=1, padding=0),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, t_emb=None):
        if t_emb is not None:
            h = x + self.t_proj(t_emb)[:, :, None, None].expand_as(x)
        else:
            h = x
        h = self.conv(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.conv1 = ConvNextBlock(in_channels, out_channels, t_dim)
        #self.conv2 = ConvNextBlock(out_channels, out_channels, t_dim)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.conv1(x, t_emb)
        #x = self.conv2(x, t_emb)
        return self.pool(x), x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.conv1 = ConvNextBlock(in_channels, out_channels, t_dim)
        #self.conv2 = ConvNextBlock(out_channels, out_channels, t_dim)
        self.unpool = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x1, x2, t_emb):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x, t_emb)
        #x = self.conv2(x, t_emb)
        return self.unpool(x)
    
    
class UNet(nn.Module):
    def __init__(self,
        in_channels=3, 
        out_channels=3, 
        hidden_channels=[64, 128, 256], 
        t_dim=256
    ):
        super().__init__()
        dims = hidden_channels

        self.downs = nn.ModuleList([
            Down(ins, outs, t_dim) for ins, outs in zip([in_channels] + dims[:-1], dims)
        ])

        mid_dim = dims[-1]
        self.mid = ConvNextBlock(mid_dim, mid_dim, t_dim)
        self.mid_unpool = nn.ConvTranspose2d(mid_dim, mid_dim, kernel_size=4, stride=2, padding=1)

        self.ups = nn.ModuleList([
            Up(ins*2, outs, t_dim) for ins, outs in zip(dims[1:][::-1], dims[:-1][::-1])
        ])

        self.final = nn.Sequential(
            ConvNextBlock(dims[0], dims[0]),
            nn.Conv2d(dims[0], out_channels, kernel_size=1)
        )

    def forward(self, x, t_emb):
        hs = []
        for down in self.downs:
            x, h = down(x, t_emb)
            hs.append(h)

        assert x.size(-1) > 1, "Too many downsamples, intermediate size is 1 x 1"
        x = self.mid(x, t_emb)
        x = self.mid_unpool(x)

        for up in self.ups:
            h = hs.pop()
            x = up(x, h, t_emb)

        return self.final(x)

class DiffusionModule(pl.LightningModule):
    def __init__(self,
        max_t: int = 1000,
        beta_1: float = 1e-4,
        beta_t: float = 1e-2,
        t_dim: int = 256,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.max_t = max_t
        self.lr = lr

        betas = np.linspace(beta_1, beta_t, self.max_t)
        self.alphas = torch.tensor(np.cumprod(1 - betas), dtype=torch.float)
        
        self.t_embed = nn.Embedding(max_t, t_dim)
        self.unet = UNet(t_dim=t_dim)
        
    def add_noise(self, x, t):
        batch_size = x.size(0)
        alphas = torch.gather(self.alphas.unsqueeze(0).expand(batch_size, -1), 1, t.unsqueeze(1))
        alphas = alphas[:, :, None, None].expand_as(x)
        
        eps = torch.randn_like(x)
        x_hat = torch.sqrt(alphas)*x + torch.sqrt(1 - alphas)*eps
        
        return x_hat, eps
        
    def forward(self, x, t):
        t_emb = self.t_embed(t)
        return self.unet(x, t_emb)

    def get_loss(self, x):
        batch_size = x.size(0)
        t = torch.randint(self.max_t, size=(batch_size,))
        x_hat, eps = self.add_noise(x, t)
        
        eps_hat = self.forward(x_hat, t)
        
        loss = (eps - eps_hat).pow(2).sum(dim=1).mean()
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch):
        return self.get_loss(batch)

    def validation_step(self, batch):
        return self.get_loss(batch)

    def test_step(self, batch):
        return self.get_loss(batch)