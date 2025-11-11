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
        
        self.ae = AutoEncoder()
        # Determine latent dimension by encoding a dummy tensor
        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.ones(1, 3, 256, 256)).shape[1]
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                               num_timesteps=num_timesteps)
    
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
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # 假设 batch 包含 (condition, target) 两个元素
        condition, target = batch
        device = condition.device
    
        # 使用 AutoEncoder 将图像编码到 latent 空间（这里不更新 AE 参数）
        with torch.no_grad():
            cond_latent = self.ae.encode(self.input_T(condition)).detach() * self.latent_scale_factor
            real_latent = self.ae.encode(self.input_T(target)).detach() * self.latent_scale_factor
    
        if optimizer_idx == 1:  # 更新判别器
            fake_latent = self.model(condition=cond_latent).detach()
            pred_real = self.discriminator(torch.cat([cond_latent, real_latent], dim=1))
            pred_fake = self.discriminator(torch.cat([cond_latent, fake_latent], dim=1))
            loss_D_real = self.gan_loss(pred_real, True)
            loss_D_fake = self.gan_loss(pred_fake, False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
    
            self.log_dict({
                'loss_D_real': loss_D_real,
                'loss_D_fake': loss_D_fake,
                'loss_D': loss_D
            }, on_step=True, on_epoch=True, prog_bar=True)
            print("Discriminator step is running")
            return loss_D
    
        elif optimizer_idx == 0:  # 更新生成器
            b = real_latent.shape[0]
            t = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=device).long()
    
            # 计算 diffusion loss
            output_noisy, noise = self.model.forward_process(real_latent, t, return_noise=True)
            model_input = torch.cat([output_noisy, cond_latent], dim=1)
            noise_hat = self.model.model(model_input, t)
            diffusion_loss = self.model.loss_fn(noise, noise_hat)
    
            # 计算生成器的 GAN loss
            fake_latent = self.model(condition=cond_latent)
            pred_fake_for_G = self.discriminator(torch.cat([cond_latent, fake_latent], dim=1))
            gan_loss = self.gan_loss(pred_fake_for_G, True) # Original code had self.gan_loss
            
            # Assuming total_loss was intended to be the sum of diffusion and GAN loss for G
            total_loss = diffusion_loss + gan_loss 
    
            self.log_dict({
                'loss_diffusion': diffusion_loss,
                'loss_G_gan': gan_loss,
                'loss_G_total': total_loss 
            }, on_step=True, on_epoch=True, prog_bar=True)
            print("Generator step is running")
            return total_loss

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
                 alpha=0.05):
        super().__init__(train_dataset, valid_dataset,
                         num_timesteps, latent_scale_factor,
                         batch_size, lr)
        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=self.latent_dim,
            condition_channels=self.latent_dim,
            num_timesteps=num_timesteps
        )
        # Initialize log_sigma_gan such that sigma = exp(log_sigma_gan) = 5.0
        # So, log_sigma_gan = log(5.0)
        self.log_sigma_gan = nn.Parameter(torch.tensor(math.log(5.0))) 
        self.gamma_r1 = gamma_r1 # Coefficient for gradient penalties (R1+R2)
        self.alpha = alpha # Scaling factor for D loss and log_sigma_gan regularization
        
        self.discriminator = NLayerDiscriminator(
            input_nc=2 * self.latent_dim, # condition_latent + sample_latent
            ndf=64,
            n_layers=3,
            norm_layer=nn.InstanceNorm2d
        )
        
        # Instantiate AdversarialTraining
        # self.model is the Generator, self.discriminator is the Discriminator
        self.adversarial_trainer = AdversarialTraining(Generator=self.model, Discriminator=self.discriminator)


    @torch.no_grad()
    def forward(self, condition, *args, **kwargs):
        # Encode condition image to latent and run diffusion model to generate output image
        condition_latent = self.ae.encode(self.input_T(condition.to(self.device))).detach() * self.latent_scale_factor
        output_latent = self.model(condition_latent, *args, **kwargs)  # latent code of output
        output_latent = output_latent / self.latent_scale_factor
        return self.output_T(self.ae.decode(output_latent))
    
    def configure_optimizers(self):
        opt_G = torch.optim.Adam(
            list(self.model.parameters()) + [self.log_sigma_gan],
            lr=self.lr
        )
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [opt_G, opt_D]
             
    def training_step(self, batch, batch_idx, optimizer_idx):
        condition_img, target_img = batch
        device = condition_img.device
        
        with torch.no_grad():
            condition_latent = (
                self.ae.encode(self.input_T(condition_img)).detach() * self.latent_scale_factor
            )
            real_latent_samples = ( # Renamed for clarity
                self.ae.encode(self.input_T(target_img)).detach() * self.latent_scale_factor
            )

        # Define preprocessor for discriminator input
        # Preprocessor combines condition_latent with sample_latent (real or fake)
        preprocessor_fn = lambda sample_latent: torch.cat([condition_latent.detach(), sample_latent], dim=1)

        # ── Generator Update (optimizer_idx == 0) ───────────────────────
        if optimizer_idx == 0:
            # Diffusion loss (primary objective for G)
            b = real_latent_samples.size(0)
            t = torch.randint(
                0, self.model.forward_process.num_timesteps, (b,), device=device
            ).long()
            output_noisy, noise = self.model.forward_process(
                real_latent_samples, t, return_noise=True
            )
            # model_input for diffusion model's UNet
            diffusion_model_input = torch.cat([output_noisy, condition_latent], dim=1)
            noise_hat = self.model.model(diffusion_model_input, t) # UNet call
            diffusion_loss = self.model.loss_fn(noise, noise_hat)

            # Adversarial loss for G (using AdversarialTraining logic)
            # self.adversarial_trainer.Generator is self.model
            # For conditional generation, Generator takes condition_latent as input (Noise argument in AdversarialTraining)
            fake_latent_G = self.adversarial_trainer.Generator(condition_latent) 
            
            # Detach real samples for G's adversarial loss calculation
            real_latent_G_detached = real_latent_samples.detach()
            
            # self.adversarial_trainer.Discriminator is self.discriminator
            # Discriminator takes preprocessed input (concatenated condition and sample)
            # The `Conditions` argument in `AdversarialTraining.Discriminator` is not used here as condition is part of preprocessed input
            fake_logits_G = self.adversarial_trainer.Discriminator(preprocessor_fn(fake_latent_G))
            real_logits_G = self.adversarial_trainer.Discriminator(preprocessor_fn(real_latent_G_detached))
            
            relativistic_logits_G = fake_logits_G - real_logits_G
            # G wants to maximize this (minimize negative) -> F.softplus(-relativistic_logits_G)
            gan_loss_G_adv = F.softplus(-relativistic_logits_G).mean()

            # Combine with diffusion loss using adaptive weighting (sigma_gan)
            sigma = torch.exp(self.log_sigma_gan)
            loss_G_total = diffusion_loss \
                           + gan_loss_G_adv / (2 * sigma**2) \
                           + self.alpha * self.log_sigma_gan # Regularization for sigma
            
            self.log_dict(
                {
                    "loss_diffusion": diffusion_loss.item(),
                    "loss_G_gan_adv_rel": gan_loss_G_adv.item(), 
                    "sigma_gan": sigma.item(),
                    "loss_G_total": loss_G_total.item(),
                },
                on_step=True, on_epoch=True, prog_bar=True
            )
            return loss_G_total
    
        # ── Discriminator Update (optimizer_idx == 1) ──────────────────
        elif optimizer_idx == 1:
            # Real samples for D - need gradients for ZCGP (R1)
            real_latent_D_grad = real_latent_samples.detach().clone().requires_grad_(True)
            
            # Generated samples (detached for D update) - need gradients for ZCGP (R2)
            with torch.no_grad():
                # self.adversarial_trainer.Generator is self.model
                fake_latent_D_no_grad = self.adversarial_trainer.Generator(condition_latent).detach()
            fake_latent_D_grad = fake_latent_D_no_grad.clone().requires_grad_(True)

            # self.adversarial_trainer.Discriminator is self.discriminator
            real_logits_D = self.adversarial_trainer.Discriminator(preprocessor_fn(real_latent_D_grad))
            fake_logits_D = self.adversarial_trainer.Discriminator(preprocessor_fn(fake_latent_D_grad))
            
            # Relativistic Discriminator loss
            # D wants to maximize (real_logits_D - fake_logits_D) -> minimize F.softplus(-(real_logits_D - fake_logits_D))
            relativistic_logits_D = real_logits_D - fake_logits_D
            loss_D_adv = F.softplus(-relativistic_logits_D).mean()

            # Zero-Centered Gradient Penalties (R1 and R2)
            # The gradient is w.r.t. real_latent_D_grad (the "data" part, not the condition)
            # `Critics` are `real_logits_D` and `fake_logits_D` respectively
            r1_penalty = self.adversarial_trainer.ZeroCenteredGradientPenalty(real_latent_D_grad, real_logits_D).mean()
            r2_penalty = self.adversarial_trainer.ZeroCenteredGradientPenalty(fake_latent_D_grad, fake_logits_D).mean()
            
            # Total D loss, scaled by self.alpha
            # self.gamma_r1 is used as Gamma for the combined (R1+R2) penalty term
            loss_D_total = self.alpha * (loss_D_adv + (self.gamma_r1 / 2.0) * (r1_penalty + r2_penalty))
    
            self.log_dict(
                {
                    "loss_D_adv_rel": loss_D_adv.item(), 
                    "r1_penalty_zc": r1_penalty.item(), 
                    "r2_penalty_zc": r2_penalty.item(), 
                    "loss_D_combined": loss_D_total.item(),
                },
                on_step=True, on_epoch=True, prog_bar=True
            )
            return loss_D_total

    
    def validation_step(self, batch, batch_idx):
        # 验证阶段仅计算原始扩散损失，不加入 GAN 对抗损失
        condition_img, target_img = batch
        condition_latent = self.ae.encode(self.input_T(condition_img)).detach() * self.latent_scale_factor
        output_latent = self.ae.encode(self.input_T(target_img)).detach() * self.latent_scale_factor
        
        # Compute diffusion loss (MSE) only for evaluation
        b = output_latent.shape[0]
        device = output_latent.device
        t = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise = self.model.forward_process(output_latent, t, return_noise=True)
        model_input = torch.cat([output_noisy, condition_latent], dim=1)
        noise_hat = self.model.model(model_input, t)
        diffusion_loss = self.model.loss_fn(noise, noise_hat)
        
        self.log('val_loss', diffusion_loss)
        return diffusion_loss



class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        # Samples: The tensor w.r.t. which gradients are computed (e.g., real or fake latents)
        # Critics: The output of the discriminator for these samples
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True, retain_graph=True)
        return Gradient.square().sum(dim=list(range(1, Gradient.ndim))) # Sum over all but batch dimension
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1, Preprocessor=lambda x: x):
        # This method is not directly called due to its internal .backward()
        # Its logic is reimplemented in LatentDiffusionConditional.training_step
        # In LatentDiffusionConditional, Noise corresponds to condition_latent, 
        # and Conditions argument here is not directly used as conditions are part of Preprocessor.
        FakeSamples = self.Generator(Noise, Conditions) 
        RealSamples = RealSamples.detach()
        
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        
        RelativisticLogits = FakeLogits - RealLogits # G wants to maximize this
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits) # softplus(-x) for minimization
        
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
        
        RelativisticLogits = RealLogits - FakeLogits # D wants to maximize this
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits) # softplus(-x) for minimization
        
        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]