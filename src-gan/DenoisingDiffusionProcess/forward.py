"""

    This file contains implementations of the forward diffusion process

    Current Models:
    
    1) Gaussian Diffusion

"""
import torch
from torch import nn

from .beta_schedules import *

class ForwardModel(nn.Module):
    """
        (Forward Model Template)
    """
    
    def __init__(self,
                 num_timesteps=1000,
                 schedule='linear'
                ):
        
        super().__init__()
        self.schedule=schedule
        self.num_timesteps=num_timesteps
     
    @torch.no_grad()
    def forward(self, x_0, t):
        """
            Get noisy sample at t given x_0
        """
        raise NotImplemented
    
    @torch.no_grad()
    def step(self, x_t, t):
        """
            Get next sample in the process
        """
        raise NotImplemented   
        

class GaussianForwardProcess(ForwardModel):
    """
        Gassian Forward Model
    """
    
    def __init__(self,
                 num_timesteps=1000,
                 schedule='linear'
                ):
        
        super().__init__(num_timesteps=num_timesteps,
                         schedule=schedule
                        )
        
        # get process parameters
        self.register_buffer('betas',get_beta_schedule(self.schedule,self.num_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        # <<< 新增的两行别名 >>>
        self.register_buffer('sqrt_alphas_cumprod',
                             self.alphas_cumprod_sqrt.clone())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             self.alphas_one_minus_cumprod_sqrt.clone())

    @torch.no_grad()
    def forward(self, x_0, t, return_noise=False):
        """
            Get noisy sample at t given x_0
            
            q(x_t | x_0)=N(x_t; alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        assert (t<self.num_timesteps).all()
        
        b=x_0.shape[0]
        mean=x_0*self.alphas_cumprod_sqrt[t].view(b,1,1,1)
        std=self.alphas_one_minus_cumprod_sqrt[t].view(b,1,1,1)
        
        noise=torch.randn_like(x_0)
        output=mean+std*noise        
        
        if not return_noise:
            return output
        else:
            return output, noise
    
    @torch.no_grad()
    def step(self, x_t, t, return_noise=False):
        """
            Get next sample in the process
            
            q(x_t | x_t-1)=N(x_t; alphas_sqrt(t)*x_0,betas(t)*I)
        """
        assert (t<self.num_timesteps).all()
        
        mean=self.alphas_sqrt[t]*x_t
        std=self.betas_sqrt[t]
        
        noise=torch.randn_like(x_t)
        output=mean+std*noise        
        
        if not return_noise:
            return output
        else:
            return output, noise

    @torch.no_grad()
    def sample_posterior(self, x0_hat, x_t_plus1, t):
        """
        Return x_t sampled from q(x_t | x_{t+1}, x0_hat)
        x0_hat       : [B,C,H,W]  - 网络预测的 x_0
        x_t_plus1    : [B,C,H,W]  - 当前 noisy (即 x_{t+1})
        t            : [B]        - int64, 与 x_{t+1} 对应的时间步
        """
        betas        = self.betas[t]                          # [B]
        alphas       = self.alphas[t]
        alphas_bar   = self.alphas_cumprod[t]
        alphas_bar_p = torch.cat(
            [alphas_bar.new_ones(1), self.alphas_cumprod[:-1]]
        )[t]                                                # \bar α_{t-1}

        # 公式系数
        coef1 = betas * torch.sqrt(alphas_bar_p) / (1. - alphas_bar)
        coef2 = (1. - alphas_bar_p) * torch.sqrt(alphas) / (1. - alphas_bar)
        coef1 = coef1.view(-1, 1, 1, 1)
        coef2 = coef2.view(-1, 1, 1, 1)

        mean = coef1 * x0_hat + coef2 * x_t_plus1
        var  = betas * (1. - alphas_bar_p) / (1. - alphas_bar)
        std  = torch.sqrt(var).view(-1, 1, 1, 1)

        eps  = torch.randn_like(x_t_plus1)
        x_t  = mean + std * eps                              # re-param 采样

        return x_t
        # -----------------------------------------------------------
    #  带梯度：计算 q(x_t | x_{t+1}, x0_hat) 的 μ、σ²
    # -----------------------------------------------------------
    def get_posterior_mean_variance(self, x0_hat, x_t_plus1, t):
        """
        Return (mean, var) of q(x_t | x_{t+1}, x0_hat) for DDPM posterior.
        不加 @torch.no_grad —— 保留梯度，供 G 分支反传。
        """
        betas       = self.betas[t]                 # β_t
        alphas      = self.alphas[t]                # α_t
        abar_t      = self.alphas_cumprod[t]        # \bar α_t
        abar_prev   = torch.cat(
            [abar_t.new_ones(1), self.alphas_cumprod[:-1]]
        )[t]                                        # \bar α_{t-1}
    
        coef1 = betas * torch.sqrt(abar_prev) / (1. - abar_t)
        coef2 = (1. - abar_prev) * torch.sqrt(alphas) / (1. - abar_t)
        coef1 = coef1.view(-1, 1, 1, 1)
        coef2 = coef2.view(-1, 1, 1, 1)
    
        mean = coef1 * x0_hat + coef2 * x_t_plus1
        var  = betas * (1. - abar_prev) / (1. - abar_t)
        var  = var.view(-1, 1, 1, 1)
        return mean, var
