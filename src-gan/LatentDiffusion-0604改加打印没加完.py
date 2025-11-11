import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.models import AutoencoderKL
from .DenoisingDiffusionProcess import *
# 新增导入：从 networks.py 引入 PatchGAN 判别器和 GANLoss
from .networks import NLayerDiscriminator
import math # For math.log

class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type="stabilityai/sd-vae-ft-ema"  # @param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                ):
        """
            A wrapper for an AutoEncoder model (using a pretrained AutoencoderKL by default).
            A custom AutoEncoder could be used with the same interface.
        """
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)
        
    def forward(self, input):
        return self.model(input).sample
    
    def encode(self, input, mode=False):
        dist = self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self, input):
        return self.model.decode(input).sample

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        """
            This is a simplified version of Latent Diffusion (unconditional).
        """
        super().__init__()

        # 图像预处理：把 latent decode 回像素，再 resize/crop 到 Inception 要求的 299×299
        self.to_img = lambda x: self.ae.decode(x / self.latent_scale_factor)
        self.resize_to_299 = T.Resize((299, 299))
        self.normalize_inception = T.Normalize((0.5,)*3, (0.5,)*3)  # 取决于 Inception 的预处理
        self.automatic_optimization = False       
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size

        self.automatic_optimization = False
        self.ae = AutoEncoder()
        # Determine latent dimension by encoding a dummy tensor
        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.ones(1, 3, 256, 256)).shape[1]
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                               num_timesteps=num_timesteps)
        # Note: self.discriminator, self.gan_loss, self.netG, self.netD are not initialized here
        # but are used in training_step and configure_optimizers.
        # This class might be a base or incomplete for standalone GAN training.
    
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        # Decode the model output (latent) back to image and transform to [0,1] range
        return self.output_T(self.ae.decode(self.model(*args, **kwargs) / self.latent_scale_factor))
    
    def input_T(self, input):
        # Transform input from [0,1] range to [-1,1] range
        return (input.clip(0, 1) * 2) - 1
    
    def output_T(self, input):
        # Transform output from [-1,1] range to [0,1] range
        return (input + 1) / 2
    
    def training_step(self, batch, batch_idx): # optimizer_idx removed
        opt_G, opt_D = self.optimizers()

        # 假设 batch 包含 (condition, target) 两个元素
        condition, target = batch
        device = condition.device
    
        # 使用 AutoEncoder 将图像编码到 latent 空间（这里不更新 AE 参数）
        with torch.no_grad():
            cond_latent = self.ae.encode(self.input_T(condition)).detach() * self.latent_scale_factor
            real_latent = self.ae.encode(self.input_T(target)).detach() * self.latent_scale_factor
    
        # ── Discriminator Update ─────────────────────────────────────────
        # (Original logic from optimizer_idx == 1)
        # （1）先随机选一个噪声步 t
        b_D = real_latent.shape[0] # Renamed to b_D to avoid clash if G part also uses b
        t_D = torch.randint(0, self.model.forward_process.num_timesteps - 1, (b_D,), device=device).long() # Renamed to t_D
    
        # （2）从真实 latent x₀ = real_latent 开始，做前向扩散到 xₜ 和 xₜ₊₁
        x_t   = self.model.forward_process(real_latent_samples, t_D,     return_noise=False) 
        x_tp1 = self.model.forward_process(real_latent_samples, t_D + 1, return_noise=False)  #might go out of bounds if t_D is T-1
    
        # （3）real_sample = xₜ；拼接条件输出供判别器
        real_noisy = x_t.detach()   # 真样本 noisy latent
        # Assuming self.discriminator and self.gan_loss are defined if this class is used for GAN training
        real_inputD = torch.cat([cond_latent, real_noisy], dim=1)
        pred_real = self.discriminator(real_inputD) 
    
        # （4）用同一个 xₜ₊₁ 生成假样本 x̂ₜ：
        with torch.no_grad():
            # 4.1 先让 UNet 预测噪声：noise_hat = model.model([xₜ₊₁, cond_latent], t)
            model_input_D = torch.cat([x_tp1.detach(), cond_latent], dim=1) # Renamed
            noise_hat_D = self.model.model(model_input_D, t_D) # Renamed
    
            # 4.2 根据后验公式 sample x̂ₜ
            x0_hat = (x_tp1 - self.model.forward_process.sqrt_one_minus_alphas_cumprod[t_D+1].view(-1,1,1,1) * noise_hat_D) \
                     / self.model.forward_process.sqrt_alphas_cumprod[t_D+1].view(-1,1,1,1)
    
            posterior_mean, posterior_var = self.model.get_posterior_mean_variance(
                x0_hat, x_tp1, t_D
            )
            posterior_std = torch.sqrt(posterior_var)
            x_t_hat = posterior_mean + posterior_std * torch.randn_like(posterior_std)
    
        fake_noisy = x_t_hat.detach()  # 假样本 noisy latent
        fake_inputD = torch.cat([cond_latent, fake_noisy], dim=1)
        pred_fake = self.discriminator(fake_inputD)
    
        # （5）计算判别器的对抗损失
        loss_D_real = self.gan_loss(pred_real, True)
        loss_D_fake = self.gan_loss(pred_fake, False)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
    
        opt_D.zero_grad()
        self.manual_backward(loss_D)
        opt_D.step()

        self.log_dict({
            'loss_D_real': loss_D_real,
            'loss_D_fake': loss_D_fake,
            'loss_D': loss_D
        }, on_step=True, on_epoch=True, prog_bar=True)

        # ── Generator Update ───────────────────────────────────────────
        # (Original logic from optimizer_idx == 0)
        b_G = real_latent.shape[0] # Renamed
        t_G = torch.randint(0, self.model.forward_process.num_timesteps - 1, (b_G,), device=device).long() # Renamed
    
        # 计算 diffusion loss
        output_noisy_G, noise_G = self.model.forward_process(real_latent, t_G, return_noise=True) # Renamed
        model_input_G = torch.cat([output_noisy_G, cond_latent], dim=1) # Renamed
        noise_hat_G = self.model.model(model_input_G, t_G) # Renamed
        diffusion_loss = self.model.loss_fn(noise_G, noise_hat_G)
    
        # 计算生成器的 GAN loss
        # self.model is DenoisingDiffusionProcess, its forward is sample. It might need condition_latent.
        # The original LatentDiffusion is unconditional, so self.model() might not take condition.
        # This part seems more suited for LatentDiffusionConditional.
        # Assuming self.model is the generator G.
        fake_latent = self.model(condition=cond_latent) # This call might need adjustment based on self.model's signature
        pred_fake_for_G = self.discriminator(torch.cat([cond_latent, fake_latent], dim=1))
        gan_loss = self.gan_loss(pred_fake_for_G, True)
        
        total_loss_G = diffusion_loss + gan_loss # Renamed
    
        opt_G.zero_grad()
        self.manual_backward(total_loss_G)
        opt_G.step()

        self.log_dict({
            'loss_diffusion': diffusion_loss,
            'loss_G_gan': gan_loss,
            'loss_G_total': total_loss_G 
        }, on_step=True, on_epoch=True, prog_bar=True)
        print("Generator step is running")
        # training_step with manual_optimization should not return the loss for backpropagation

    def validation_step(self, batch, batch_idx):
        # Encode validation images to latent space and compute diffusion loss
        latents = self.ae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        self.log('val_loss', loss)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None
    
#    def configure_optimizers(self):
#        return torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
    def configure_optimizers(self):
        # 假设 self.netG, self.netD 是你定义好的网络
        # If self.model is G and self.discriminator is D:
        # opt_G = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        # The current code uses self.netG and self.netD which are not initialized in __init__
        # This will raise an AttributeError if self.netG or self.netD are not set elsewhere.
        opt_G = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        return [opt_G, opt_D], []

class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4,
                 gamma_r1=10.0, # This will be used as Gamma for ZCGP (R1+R2)
                 alpha=0.05,
                 debug=False, **kwargs):
        super().__init__(train_dataset, valid_dataset,
                         num_timesteps, latent_scale_factor,
                         batch_size, lr,**kwargs)
        self.debug = debug              # <<<<<< 新增
        self.automatic_optimization = False
        self.model = DenoisingDiffusionConditionalProcess( # This is the Generator (G)
            generated_channels=self.latent_dim,
            condition_channels=self.latent_dim,
            num_timesteps=num_timesteps
        )
        self.time_embedding = nn.Embedding(num_timesteps, self.latent_dim)
        self.log_sigma_gan = nn.Parameter(torch.tensor(math.log(5.0))) 
        self.gamma_r1 = gamma_r1 
        self.alpha = alpha 
        
        self.discriminator = NLayerDiscriminator( # This is the Discriminator (D)
            input_nc=3 * self.latent_dim, 
            ndf=128,
            n_layers=4,
            norm_layer=nn.InstanceNorm2d
        )
        self.adversarial_trainer = AdversarialTraining(Generator=self.model, Discriminator=self.discriminator)
        if self.debug:
            def print_forward(name):
                def hook(_, inp, out):
                    _stat(f"{name}.inp", inp[0])
                    _stat(f"{name}.out", out)
                return hook
        
            self.discriminator.register_forward_hook(print_forward("D"))
            self.model.model.register_forward_hook(print_forward("G_UNet"))

    @torch.no_grad()
    def forward(self, condition, *args, **kwargs):
        condition_latent = self.ae.encode(self.input_T(condition.to(self.device))).detach() * self.latent_scale_factor
        output_latent = self.model(condition_latent, *args, **kwargs) 
        output_latent = output_latent / self.latent_scale_factor
        return self.output_T(self.ae.decode(output_latent))

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(
            list(self.model.parameters()) + [self.log_sigma_gan], # model is G
            lr=self.lr
        )
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr) # discriminator is D
        return [opt_G, opt_D] # opt_G is optimizers()[0], opt_D is optimizers()[1]
             
    def training_step(self, batch, batch_idx): # optimizer_idx removed
        opt_G, opt_D = self.optimizers()
        
        condition_img, target_img = batch
        device = condition_img.device
        
        with torch.no_grad():
            condition_latent = (
                self.ae.encode(self.input_T(condition_img)).detach() * self.latent_scale_factor
            )
            real_latent_samples = ( 
                self.ae.encode(self.input_T(target_img)).detach() * self.latent_scale_factor
            )

        # ── Discriminator Update (original logic from optimizer_idx == 1) ──────────────────
        # （1）随机选 t ∈ [0, T-2]
        b_D = real_latent_samples.size(0)
        t_D = torch.randint(0, self.model.forward_process.num_timesteps - 1,
                          (b_D,), device=device).long()
    
        # （2）真实 forward diffusion → x_t, x_{t+1}
        x_t   = self.model.forward_process(real_latent_samples, t_D,     return_noise=False) 
        x_tp1 = self.model.forward_process(real_latent_samples, t_D + 1, return_noise=False) 
    
        # （3）构造“真 noisy latent” real_noisy，并保留 grad 用于 R1
        real_noisy = x_t.detach().clone().requires_grad_(True)
    
        # （4）构造“伪 noisy latent” fake_noisy via UNet + 后验采样
        with torch.no_grad():
            # 4.1 UNet 预测 noise_hat
            model_input_D = torch.cat([x_tp1.detach(), condition_latent], dim=1)
            noise_hat_D  = self.model.model(model_input_D, t_D)
            sqrt_1m_cum  = self.model.forward_process.sqrt_one_minus_alphas_cumprod[t_D + 1].view(-1,1,1,1)
            sqrt_cum     = self.model.forward_process.sqrt_alphas_cumprod[t_D + 1].view(-1,1,1,1)
            x0_hat_D     = (x_tp1 - sqrt_1m_cum * noise_hat_D) / sqrt_cum
            
            # 4.2 用 forward_process 内置采样，直接得到 fake_noisy
            fake_noisy   = self.model.forward_process.sample_posterior(x0_hat_D.detach(),
                                                                       x_tp1.detach(),
                                                                       t_D)
        fake_noisy = fake_noisy.detach().clone().requires_grad_(True)
    
        # （5）time-conditioning
        t_embed_D = self.time_embedding(t_D)      
        H_D, W_D    = real_noisy.shape[2], real_noisy.shape[3]
        t_map_D   = t_embed_D.view(-1, self.latent_dim, 1, 1).expand(-1, -1, H_D, W_D)
    
        # （6）拼接 condition_latent ∥ noisy_latent ∥ t_map 送给判别器
        real_inputD = torch.cat([condition_latent, real_noisy, t_map_D], dim=1)
        fake_inputD = torch.cat([condition_latent, fake_noisy, t_map_D], dim=1)
    
        real_logits_D = self.adversarial_trainer.Discriminator(real_inputD)
        fake_logits_D = self.adversarial_trainer.Discriminator(fake_inputD)
    
        # （7）Relativistic Loss
        relativistic_logits_D = real_logits_D - fake_logits_D
        loss_D_adv = F.softplus(-relativistic_logits_D).mean()
    
        # （8）R1 / R2 零中心梯度惩罚
        r1_penalty = self.adversarial_trainer.ZeroCenteredGradientPenalty(real_noisy, real_logits_D).mean()
        r2_penalty = self.adversarial_trainer.ZeroCenteredGradientPenalty(fake_noisy, fake_logits_D).mean()
    
        # （9）总 D 损失
        loss_D_total = self.alpha * (
            loss_D_adv + (self.gamma_r1 / 2.0) * (r1_penalty + r2_penalty)
        )
        
        opt_D.zero_grad()
        self.manual_backward(loss_D_total)
        opt_D.step()

        self.log_dict({
            "loss_D_adv_rel": loss_D_adv.item(),
            "r1_penalty_zc": r1_penalty.item(),
            "r2_penalty_zc": r2_penalty.item(),
            "loss_D_combined": loss_D_total.item(),
        }, on_step=True, on_epoch=True, prog_bar=True)

        # ── Generator Update (original logic from optimizer_idx == 0) ───────────────────────
        # Diffusion loss (primary objective for G)
        b_G = real_latent_samples.size(0)
        t_G = torch.randint(
            0, self.model.forward_process.num_timesteps - 1, (b_G,), device=device
        ).long()
        output_noisy_G, noise_G = self.model.forward_process(
            real_latent_samples, t_G, return_noise=True
        )
        diffusion_model_input_G = torch.cat([output_noisy_G, condition_latent], dim=1)
        noise_hat_G = self.model.model(diffusion_model_input_G, t_G) 
        diffusion_loss = self.model.loss_fn(noise_G, noise_hat_G)

        # Adversarial loss for G
        # fake_latent_G = self.adversarial_trainer.Generator(condition_latent) # This was for a simpler GAN loss
        # The following is the LDM-specific GAN loss for G:
        
        real_latent_G_detached = real_latent_samples.detach()

        t2_G = t_G # Reuse t from diffusion loss calculation for G's adversarial part
        x_t2_G   = self.model.forward_process(real_latent_G_detached, t2_G,   return_noise=False)
        x_t2p1_G = self.model.forward_process(real_latent_G_detached, t2_G + 1, return_noise=False)
    
        real_noisy_G = x_t2_G.detach()
    
        with torch.no_grad(): # This UNet call should not affect G's gradients for diffusion loss
            input_unet_G = torch.cat([x_t2p1_G.detach(), condition_latent], dim=1)
            noise_hat_for_G_adv = self.model.model(input_unet_G, t2_G) # Renamed
    
            sqrt_1m_cum_G = self.model.forward_process.sqrt_one_minus_alphas_cumprod[t2_G + 1].view(-1, 1, 1, 1)
            sqrt_cum_G    = self.model.forward_process.sqrt_alphas_cumprod[t2_G + 1].view(-1, 1, 1, 1)
            x0_hat_for_G_adv = (x_t2p1_G - sqrt_1m_cum_G * noise_hat_for_G_adv) / sqrt_cum_G # Renamed
    
            post_mean_G_adv, post_var_G_adv = self.model.forward_process.get_posterior_mean_variance(x0_hat_for_G_adv, x_t2p1_G, t2_G) # Renamed
            post_std_G_adv = torch.sqrt(post_var_G_adv) # Renamed

        input_unet_G_grad = torch.cat([x_t2p1_G.detach(), condition_latent], dim=1) # x_t2p1_G is from real, so detach.
        noise_hat_for_G_adv_grad = self.model.model(input_unet_G_grad, t2_G) # THIS IS THE KEY: UNet call for G loss

        x0_hat_for_G_adv_grad = (x_t2p1_G.detach() - sqrt_1m_cum_G * noise_hat_for_G_adv_grad) / sqrt_cum_G
    
        post_mean_G_adv_grad, post_var_G_adv_grad = \
            self.model.forward_process.get_posterior_mean_variance(
                x0_hat_for_G_adv_grad, x_t2p1_G.detach(), t2_G
            )
        post_std_G_adv_grad = torch.sqrt(post_var_G_adv_grad)
        # Sample fake_noisy_G. Ensure randomness is fixed or detached if it shouldn't affect G's gradients.
        # Typically, the random noise is treated as a constant for a given step.
        fake_noisy_G_adv = post_mean_G_adv_grad + post_std_G_adv_grad * torch.randn_like(post_std_G_adv_grad).detach()


        t_embed_G = self.time_embedding(t2_G)
        H_G, W_G      = real_noisy_G.shape[2], real_noisy_G.shape[3] # Use H,W from real_noisy_G for consistency
        t_map_G   = t_embed_G.view(-1, self.latent_dim, 1, 1).expand(-1, -1, H_G, W_G)
    
        real_input_G = torch.cat([condition_latent, real_noisy_G, t_map_G], dim=1).detach() # D input should be detached for G loss
        fake_input_G = torch.cat([condition_latent, fake_noisy_G_adv, t_map_G], dim=1) # fake_noisy_G_adv is differentiable
    
        fake_logits_G = self.adversarial_trainer.Discriminator(fake_input_G)
        real_logits_G = self.adversarial_trainer.Discriminator(real_input_G) # D is frozen here
    
        relativistic_logits_G = fake_logits_G - real_logits_G # G wants to maximize this (minimize -relativistic_logits_G)
        gan_loss_G_adv = F.softplus(-relativistic_logits_G).mean()
    
        sigma = torch.exp(self.log_sigma_gan)
        loss_G_total = diffusion_loss \
                       + gan_loss_G_adv / (2 * sigma**2) \
                       + self.alpha * self.log_sigma_gan
        
        opt_G.zero_grad()
        self.manual_backward(loss_G_total)
        opt_G.step()

        self.log_dict({
            "loss_diffusion": diffusion_loss.item(),
            "loss_G_gan_adv_rel": gan_loss_G_adv.item(),
            "sigma_gan": sigma.item(),
            "loss_G_total": loss_G_total.item(),
        }, on_step=True, on_epoch=True, prog_bar=True)
        # training_step with manual_optimization should not return the loss

    def validation_step(self, batch, batch_idx):
        condition_img, target_img = batch
        condition_latent = self.ae.encode(self.input_T(condition_img)).detach() * self.latent_scale_factor
        output_latent = self.ae.encode(self.input_T(target_img)).detach() * self.latent_scale_factor
        
        b = output_latent.shape[0]
        device = output_latent.device
        t = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise = self.model.forward_process(output_latent, t, return_noise=True)
        model_input = torch.cat([output_noisy, condition_latent], dim=1)
        noise_hat = self.model.model(model_input, t)
        diffusion_loss = self.model.loss_fn(noise, noise_hat)
        
        self.log('val_loss', diffusion_loss)
        return diffusion_loss
        
    def _stat(name, tensor):
    if tensor is None: 
        print(f"{name}: None")
        return
    print(f"{name:>20s} | shape={tuple(tensor.shape)} "
          f"mean={tensor.mean():.4f}  std={tensor.std():.4f} "
          f"min={tensor.min():.4f}  max={tensor.max():.4f}")
    def on_after_backward(self):
        if not self.debug: 
            return
        # 只打印几个关键层
        for n, p in self.model.named_parameters():
            if p.grad is not None and ("model.0" in n or ".bias" in n) :
                _stat(f"grad {n}", p.grad)
                break


class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True, retain_graph=True)
        return Gradient.square().sum(dim=list(range(1, Gradient.ndim))) 
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1, Preprocessor=lambda x: x):
        # This method is not directly called due to its internal .backward()
        # Its logic is reimplemented in LatentDiffusionConditional.training_step
        FakeSamples = self.Generator(Noise, Conditions) 
        RealSamples = RealSamples.detach()
        
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        
        RelativisticLogits = FakeLogits - RealLogits 
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits) 
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1, Preprocessor=lambda x: x):
        # This method is not directly called due to its internal .backward()
        # Its logic is reimplemented in LatentDiffusionConditional.training_step
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = self.Generator(Noise, Conditions).detach().requires_grad_(True)
        
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        
        R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
        R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
        
        RelativisticLogits = RealLogits - FakeLogits 
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits) 
        
        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]