import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class ImageSampler(nn.Module):
    def __init__(self, model: nn.Module, max_iter=None, channels=3, size=(32, 32)):
        super().__init__()
        
        if max_iter is None:
            max_iter = model.max_t
        
        self.model = model
        self.max_t = model.max_t
        self.max_iter = min(self.max_t, max_iter)
        self.channels = channels
        self.size = size
        
        t_schedule = torch.linspace(0, self.max_t-1, self.max_iter).round().long()
        schedule = torch.stack([torch.arange(self.max_iter), t_schedule], dim=-1).unsqueeze(-1)
        
        betas = self.model.betas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        if self.max_iter < self.max_t:
            alphas_cumprod = alphas_cumprod.gather(-1, t_schedule)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
            betas = 1 - alphas_cumprod / alphas_cumprod_prev
            posterior_variance = (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * betas
        else:
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('schedule', schedule)

    def p_mean(self, x_t, t, i):
        beta = extract(self.betas, i, x_t.shape)
        alpha_bar = extract(self.alphas_cumprod, i, x_t.shape)
        
        eps = self.model(x_t, t)
        mean = (x_t - beta/torch.sqrt(1 - alpha_bar)*eps) / torch.sqrt(1. - beta)
        return mean

    def p_sample(self, x, step):
        i, t = step
        
        mean = self.p_mean(x, t, i)
        noise = torch.randn_like(x)
        var = extract(self.posterior_variance, i, x.shape)
        
        mask = (t != 0)
        return mean + mask * torch.sqrt(var) * noise

    @torch.inference_mode()
    def sample(self, n_samples=1, device=None):
        self.model.eval()

        if device is None:
            device = self.model.device

        x = torch.randn(n_samples, self.channels, *self.size, device=device)
        for step in tqdm(self.schedule.flip(0), desc='Generating image samples'):
            x = self.p_sample(x, step)
        return x