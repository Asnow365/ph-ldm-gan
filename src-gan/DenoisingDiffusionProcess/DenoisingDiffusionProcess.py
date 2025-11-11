import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.unet_convnext import *

# 新增导入：从pix2pix的 networks.py 引入 GANLoss 和 NLayerDiscriminator
from .networks import GANLoss, NLayerDiscriminator


class DenoisingDiffusionProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,              
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels = generated_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn
        
        # Forward Process Used for Training
        self.forward_process = GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                      schedule=schedule)
        self.model = UnetConvNextBlock(dim=64,
                                       dim_mults=(1,2,4,8),
                                       channels=self.generated_channels,
                                       out_dim=self.generated_channels,
                                       with_time_emb=True)
       
        # defaults to a DDPM sampler if None is provided
        self.sampler = DDPM_Sampler(num_timesteps=self.num_timesteps) if sampler is None else sampler
        
    @torch.no_grad()
    def forward(self,
                shape=(256,256),
                batch_size=1,
                sampler=None,
                verbose=False
               ):
        """
            forward() function triggers a complete inference cycle
            
            A custom sampler can be provided as an argument!
        """           
        
        # read dimensions
        b, c, h, w = batch_size, self.generated_channels, *shape
        device = next(self.model.parameters()).device
        
        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps 
        it = reversed(range(0, num_timesteps))    
        
        x_t = torch.randn([b, self.generated_channels, h, w], device=device)
                
        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t = self.model(x_t, t)  # prediction of noise
            x_t = sampler(x_t, t, z_t)  # prediction of next state
            
        return x_t
        
    def p_loss(self, output):
        """
            Assumes output is in [-1,+1] range.
            For unconditional diffusion.
        """        
        b, c, h, w = output.shape
        device = output.device
        
        # loss for training
        
        # input is the optional condition (if any, but here unconditional)
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise = self.forward_process(output, t, return_noise=True)        

        # reverse pass
        noise_hat = self.model(output_noisy, t) 

        # apply loss
        return self.loss_fn(noise, noise_hat)
    
    
class DenoisingDiffusionConditionalProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,
                 condition_channels=3,
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn
        
        # Forward Process
        self.forward_process = GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                      schedule=schedule)
        
        # Neural Network Backbone: 接收拼接后的latent（generated + condition）
        self.model = UnetConvNextBlock(dim=64,
                                       dim_mults=(1,2,4,8),
                                       channels=self.generated_channels + condition_channels,
                                       out_dim=self.generated_channels,
                                       with_time_emb=True)
        
        # defaults to a DDPM sampler if None is provided
        self.sampler = DDPM_Sampler(num_timesteps=self.num_timesteps) if sampler is None else sampler
        
        # ===== 新增：在 latent 空间中进行对抗训练的判别器和GAN损失 =====
        # 判别器输入通道数 = generated_channels + condition_channels
        self.discriminator = NLayerDiscriminator(input_nc=self.generated_channels + self.condition_channels,
                                                   ndf=64,
                                                   n_layers=3,
                                                   norm_layer=nn.InstanceNorm2d)
        # 选择对抗损失模式：可选 'vanilla'（BCEWithLogitsLoss）或 'lsgan'
        self.gan_loss = GANLoss(gan_mode='vanilla')
        
    # 注意：为了训练时允许梯度传递，我们移除 @torch.no_grad() 装饰器
    def forward(self, condition, sampler=None, verbose=False):
        """
            forward() function triggers a complete inference cycle for conditional diffusion.
            condition: the condition image (or its latent representation) expected to have shape [B, C, H, W]
        """
        
        # read dimensions from condition
        b, c, h, w = condition.shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)
        
        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps 
        it = reversed(range(0, num_timesteps))        
        
        x_t = torch.randn([b, self.generated_channels, h, w], device=device)
                
        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            # 拼接生成部分和条件信息
            model_input = torch.cat([x_t, condition], 1).to(device)
            z_t = self.model(model_input, t)  # prediction of noise            
            x_t = sampler(x_t, t, z_t)  # prediction of next state
            
        return x_t
        
    def p_loss(self, output, condition):
        """
            Compute the combined loss for conditional diffusion:
            - Diffusion loss: MSE loss between true noise and predicted noise.
            - GAN loss: Adversarial loss in latent space; discriminator should classify (condition, output) as real.
            Assumes both output and condition are in [-1,1] range.
        """        
        b, c, h, w = output.shape
        device = output.device
        
        # Diffusion loss部分
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise = self.forward_process(output, t, return_noise=True)        
        model_input = torch.cat([output_noisy, condition], 1).to(device)
        noise_hat = self.model(model_input, t) 
        diffusion_loss = self.loss_fn(noise, noise_hat)
        
        # GAN loss部分：判别器输入为拼接后的 (condition, output)
        gan_input = torch.cat([condition, output], 1).to(device)
        pred = self.discriminator(gan_input)
        # 生成器目标：希望判别器判断假latent为真实（标签True）
        gan_loss = self.gan_loss(pred, True)
        
        # λ为GAN loss权重，建议初始取0.1
        lambda_gan = 0.1
        total_loss = diffusion_loss + lambda_gan * gan_loss
        return total_loss
