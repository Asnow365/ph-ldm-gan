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
# Ensure ForwardDiffusion is accessible, e.g., from .DenoisingDiffusionProcess import ForwardDiffusion
from .DenoisingDiffusionProcess import DenoisingDiffusionProcess, DenoisingDiffusionConditionalProcess, ForwardDiffusion 
from .networks import NLayerDiscriminator # Original Discriminator, may not be used by LDDGAN part
import math 

# Utility function (adapted from LDDGAN's diffusion.py for DDPM schedules)
def extract(input_tensor, t, shape):
    # input_tensor: 1D schedule tensor (e.g., alphas_cumprod) [num_timesteps]
    # t: batch of timesteps [batch_size]
    # shape: target shape for broadcasting, e.g., x_start.shape [batch_size, C, H, W]
    
    if input_tensor.ndim == 1:
        out = torch.gather(input_tensor, 0, t.long().to(input_tensor.device))
        # Reshape for broadcasting: [batch_size, 1, 1, ...]
        reshape_dims = [shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape_dims)
    else:
        # Fallback for unexpected input_tensor shape, though DDPM schedules are typically 1D.
        # This was closer to LDDGAN's original extract for potentially multi-dim inputs.
        _out = torch.gather(input_tensor, 0, t.long().to(input_tensor.device)) 
        reshape_dims = [shape[0]] + [1] * (len(shape) - 1) 
        return _out.reshape(*reshape_dims)


class LDDGANPosteriorCoefficients:
    def __init__(self, ddpm_forward_process: ForwardDiffusion, device):
        # ddpm_forward_process should have attributes like betas, alphas_cumprod
        self.betas = ddpm_forward_process.betas.to(device)
        self.alphas = (1. - self.betas).to(device) # alphas_t = 1 - beta_t
        self.alphas_cumprod = ddpm_forward_process.alphas_cumprod.to(device)
        
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        
        # Calculations based on LDDGAN's Posterior_Coefficients, adapted for DDPM variables
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod.clamp(min=1e-20)) 

        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20) # Ensure non-negative

        self.posterior_mean_coef1 = ( # Coef for x_0_pred
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod.clamp(min=1e-20)))
        self.posterior_mean_coef2 = ( # Coef for x_tp1 (the more noisy sample)
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod.clamp(min=1e-20)))
            
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)


def lddgan_sample_posterior(coefficients: LDDGANPosteriorCoefficients, x_0_pred, x_tp1, t_current):
    # Samples x_t_hat from p(x_t | x_{t+1}, x_0_pred) using LDDGAN formula structure
    # t_current is the timestep for x_t_hat (the less noisy sample we want to generate)
    # x_tp1 is x at t_current+1 (the more noisy sample, LDDGAN calls this x_t in its formula)
    # x_0_pred is the predicted clean sample (output of the generator UNet)
    
    mean = (
        extract(coefficients.posterior_mean_coef1, t_current, x_tp1.shape) * x_0_pred
        + extract(coefficients.posterior_mean_coef2, t_current, x_tp1.shape) * x_tp1 
    )
    log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t_current, x_tp1.shape)
    
    noise = torch.randn_like(x_tp1)
    # The LDDGAN nonzero_mask (1-(t==0)) was for their p_sample loop.
    # Here, if posterior_variance[0] (for t_current=0) is zero, no noise is added.
    # This is handled by posterior_variance calculation if beta[0]=0 leads to alpha_cumprod[0]=1.
    variance = torch.exp(log_var_clipped)
    sample_x_t_hat = mean + torch.sqrt(variance.clamp(min=1e-20)) * noise # Clamp variance just in case
    return sample_x_t_hat

def ddpm_q_sample_pairs(ddpm_forward_process: ForwardDiffusion, x_start, t, noise=None):
    # t is for x_t. So x_{t+1} uses t+1.
    # Clamp t and t+1 to be valid indices for q_sample.
    max_allowable_t_for_q_sample = ddpm_forward_process.num_timesteps - 1
    
    t_for_xt = t.clamp(0, max_allowable_t_for_q_sample)
    t_for_xtp1 = (t + 1).clamp(0, max_allowable_t_for_q_sample)

    if noise is None:
        common_noise = torch.randn_like(x_start)
    else:
        common_noise = noise

    x_t   = ddpm_forward_process.q_sample(x_start, t_for_xt, noise=common_noise)
    x_t_plus_1 = ddpm_forward_process.q_sample(x_start, t_for_xtp1, noise=common_noise)
    
    return x_t, x_t_plus_1


def get_sinusoidal_positional_embedding(timesteps, embedding_dim):
    # timesteps: [B] tensor of integer timesteps
    # embedding_dim: target dimension for the embedding
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :] 
    embedding = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: 
        embedding = F.pad(embedding, (0, 1, 0, 0)) # Pad if embedding_dim is odd
    return embedding

class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_embedding_dim, mlp_hidden_dim, mlp_output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()
        # sinusoidal_embedding_dim is the dimension of the raw sinusoidal projection
        self.sinusoidal_embedding_dim = sinusoidal_embedding_dim
        self.main = nn.Sequential(
            nn.Linear(sinusoidal_embedding_dim, mlp_hidden_dim),
            act,
            nn.Linear(mlp_hidden_dim, mlp_output_dim), # mlp_output_dim is the final t_emb dim
        )

    def forward(self, timesteps_int): # timesteps_int is the raw integer timesteps [B]
        sinusoidal_emb = get_sinusoidal_positional_embedding(timesteps_int, self.sinusoidal_embedding_dim)
        processed_emb = self.main(sinusoidal_emb)
        return processed_emb

class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, effective_emb_dim=128, downsample=False, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.downsample = downsample
        self.act = act

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        
        self.dense_emb = nn.Linear(effective_emb_dim, out_channel) 
        
        if downsample:
            self.skip_conv = nn.Conv2d(in_channel, out_channel, 1, padding=0, bias=False, stride=2)
            self.downsampler = nn.AvgPool2d(2,2) # For features
        elif in_channel != out_channel:
            self.skip_conv = nn.Conv2d(in_channel, out_channel, 1, padding=0, bias=False)
        else:
            self.skip_conv = nn.Identity()


    def forward(self, x, combined_emb): 
        h = self.act(x)
        h = self.conv1(h)
        
        emb_out = self.dense_emb(combined_emb)
        h = h + emb_out[:, :, None, None] 

        h = self.act(h)
        
        if self.downsample:
            h = self.downsampler(h)
            x_skip_input = x # x is before downsampling for skip
        else:
            x_skip_input = x

        h = self.conv2(h)
        
        x_skip = self.skip_conv(x_skip_input)
        out = (h + x_skip) / math.sqrt(2) 
        return out

class LDDGANStyleDiscriminator(nn.Module):
    def __init__(self, latent_dim, ngf=64, 
                 time_sinusoidal_emb_dim=128, time_mlp_output_dim=128, # For time embedding
                 condition_latent_dim=None, cond_mlp_output_dim=128, # For condition embedding
                 act=nn.LeakyReLU(0.2), num_conv_blocks=4): # num_conv_blocks replaces num_layers
        super().__init__()
        self.act = act
        self.condition_latent_dim = condition_latent_dim
        input_nc_to_convs = latent_dim * 2 # Concatenated x_current_sample and x_prev_sample_tp1
        
        self.t_embed = TimestepEmbedding(
            sinusoidal_embedding_dim=time_sinusoidal_emb_dim, 
            mlp_hidden_dim=time_mlp_output_dim * 4, # Example, can be tuned
            mlp_output_dim=time_mlp_output_dim, 
            act=act
        )
        effective_emb_dim = time_mlp_output_dim # Start with time embedding dim
        
        if self.condition_latent_dim is not None:
            # Assuming condition_latent is already [B, condition_latent_dim] after pooling
            self.cond_embed_mlp = nn.Sequential(
                nn.Linear(self.condition_latent_dim, cond_mlp_output_dim * 4),
                act,
                nn.Linear(cond_mlp_output_dim*4, cond_mlp_output_dim)
            )
            effective_emb_dim += cond_mlp_output_dim
        else:
            self.cond_embed_mlp = None

        self.start_conv = nn.Conv2d(input_nc_to_convs, ngf, 3, padding=1)
        current_channels = ngf
        
        self.blocks = nn.ModuleList()
        for i in range(num_conv_blocks):
            out_c = ngf * (2**(i+1)) if ngf * (2**(i+1)) <= 512 else 512 # Cap channels
            # Alternate downsampling
            use_downsample = (i % 2 == 1) and (i > 0) # e.g. Block 1 (idx 1), Block 3 (idx 3) downsample
            self.blocks.append(DownConvBlock(current_channels, out_c, 
                                             effective_emb_dim=effective_emb_dim, 
                                             downsample=use_downsample, act=act))
            current_channels = out_c
            
        self.stddev_group = 4 
        self.stddev_feat = 1 # Minibatch stddev features to add
        
        self.final_conv = nn.Conv2d(current_channels + self.stddev_feat, current_channels, 3, padding=1)
        self.final_linear = nn.Linear(current_channels, 1)

    def forward(self, x_current_sample, time_t_int, x_prev_sample_tp1, condition_latent_pooled=None):
        # x_current_sample, x_prev_sample_tp1: [B, latent_dim, H, W]
        # time_t_int: [B] tensor of integer timesteps
        # condition_latent_pooled: [B, condition_latent_dim] (already pooled if spatial)
        
        # 1. Prepare embeddings
        final_t_emb = self.t_embed(time_t_int) # [B, time_mlp_output_dim]
        combined_emb_list = [final_t_emb]

        if self.cond_embed_mlp is not None and condition_latent_pooled is not None:
            final_c_emb = self.cond_embed_mlp(condition_latent_pooled) # [B, cond_mlp_output_dim]
            combined_emb_list.append(final_c_emb)
        
        final_combined_emb = torch.cat(combined_emb_list, dim=1) # [B, effective_emb_dim]

        # 2. Concatenate image inputs
        h = torch.cat([x_current_sample, x_prev_sample_tp1], dim=1)
        
        # 3. Pass through network
        h = self.start_conv(h)
        for block_idx, block in enumerate(self.blocks):
            h = block(h, final_combined_emb)
        
        # Minibatch Standard Deviation
        batch, channel, height, width = h.shape
        group = min(batch, self.stddev_group)
        if batch > 0 and batch % group != 0: 
            group = 1 
            if batch > 1 and self.stddev_group <= batch :
                 for g_try in range(min(batch, self.stddev_group), 0, -1):
                     if batch % g_try == 0:
                         group = g_try; break
        
        if batch > 0 : # only apply if batch > 0
            stddev_input = h.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
            stddev = torch.sqrt(stddev_input.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2) 
            stddev = stddev.repeat(group, 1, height, width) 
            h = torch.cat([h, stddev], 1)
        else: # If batch is 0, stddev cannot be computed; pass h through but final_conv needs matching channels
             # This case should ideally not happen in training.
             # Add dummy features if h is empty to match final_conv input channels
             dummy_stddev = torch.zeros(0, self.stddev_feat, height, width, device=h.device, dtype=h.dtype)
             h = torch.cat([h, dummy_stddev],1)


        h = self.final_conv(h)
        h = self.act(h)
        
        if h.shape[0] > 0: # only sum if batch > 0
            h = h.sum(dim=[2,3]) 
        else: # if batch is 0, h is empty, create empty tensor for linear layer
            h = torch.zeros(0, channel, device=h.device, dtype=h.dtype) # channel here is `current_channels` before stddev

        out = self.final_linear(h) 
        return out

# ... existing AutoEncoder class ...
class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type="stabilityai/sd-vae-ft-ema"
                ):
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

# ... existing LatentDiffusion class (mostly unchanged, configure_optimizers might differ if it had GAN) ...
class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1, # VAE scale factor for latents
                 batch_size=1,
                 lr=1e-4):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size
        
        self.ae = AutoEncoder()
        with torch.no_grad():
            # Determine latent dimension by encoding a dummy tensor
            # Ensure dummy tensor matches expected VAE input (e.g. 3 channels, H, W)
            dummy_input = torch.ones(1, 3, 256, 256) # Example size, adjust if VAE expects different
            if hasattr(self.ae.model.config, 'latent_channels'):
                 self.latent_dim = self.ae.model.config.latent_channels
            else: # Fallback by encoding
                 self.latent_dim = self.ae.encode(dummy_input).shape[1]


        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                               num_timesteps=num_timesteps)
        
        # Image transformation for metrics (if needed)
        self.to_img = lambda x: self.ae.decode(x / self.latent_scale_factor) # From latent to image
        self.resize_to_299 = T.Resize((299, 299))
        self.normalize_inception = T.Normalize((0.5,)*3, (0.5,)*3)

    @torch.no_grad()
    def forward(self, *args, **kwargs): # For sampling
        # Decode the model output (latent) back to image and transform to [0,1] range
        # self.model here is DenoisingDiffusionProcess, its forward is sample()
        return self.output_T(self.ae.decode(self.model(*args, **kwargs) / self.latent_scale_factor))
    
    def input_T(self, input_pixel_space): # From [0,1] to [-1,1] for VAE
        return (input_pixel_space.clip(0, 1) * 2) - 1
    
    def output_T(self, output_pixel_space): # From [-1,1] (VAE output) to [0,1]
        return (output_pixel_space.clip(-1,1) + 1) / 2
    
    def training_step(self, batch, batch_idx): # Removed optimizer_idx for unconditional
        # For unconditional LatentDiffusion, batch is typically just the target images
        if isinstance(batch, (list, tuple)): # If dataset returns (img, label) or similar
            target_img = batch[0]
        else:
            target_img = batch

        device = target_img.device
    
        with torch.no_grad():
            real_latent_samples = self.ae.encode(self.input_T(target_img)).detach() * self.latent_scale_factor
    
        b = real_latent_samples.shape[0]
        t = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=device).long()
    
        output_noisy, noise_target = self.model.forward_process(real_latent_samples, t, return_noise=True)
        # Unconditional UNet model takes (noisy_sample, time)
        noise_hat = self.model.model(output_noisy, t) 
        diffusion_loss = self.model.loss_fn(noise_target, noise_hat)
            
        self.log_dict({
            'loss_diffusion': diffusion_loss.item(),
        }, on_step=True, on_epoch=True, prog_bar=True)
        return diffusion_loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            val_img = batch[0]
        else:
            val_img = batch
        latents = self.ae.encode(self.input_T(val_img)).detach() * self.latent_scale_factor
        # p_loss in DenoisingDiffusionProcess calculates loss for validation
        loss = self.model.p_loss(latents, loss_fn_type='l1') # Or other loss type
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return None
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr_g=1e-4, # Renamed lr to lr_g for Generator (UNet)
                 # LDDGAN specific params
                 lr_d_lddgan=1e-4, 
                 r1_gamma_lddgan=0.1, # LDDGAN used 0.05 or 0.1, adjust as needed
                 gan_loss_weight_g=0.1, # Weight for LDDGAN GAN loss for G, adjust as needed
                 rec_loss_weight_g=1.0, 
                 d_ngf_lddgan=64,      
                 d_num_conv_blocks_lddgan=4,
                 d_time_sinusoidal_emb_dim=128,
                 d_time_mlp_output_dim=128,
                 d_cond_mlp_output_dim=128
                ):
        # Call LatentDiffusion init but override LR for G
        super().__init__(train_dataset, valid_dataset,
                         num_timesteps, latent_scale_factor,
                         batch_size, lr=lr_g) 
        
        # Diffusion model (Generator for diffusion loss)
        # DenoisingDiffusionConditionalProcess's UNet (self.model.model) will be G
        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=self.latent_dim,
            condition_channels=self.latent_dim, # For concatenating condition_latent to UNet input
            num_timesteps=num_timesteps
        )
        
        # LDDGAN Style Discriminator
        self.discriminator_lddgan = LDDGANStyleDiscriminator(
            latent_dim=self.latent_dim,
            ngf=d_ngf_lddgan,
            time_sinusoidal_emb_dim=d_time_sinusoidal_emb_dim,
            time_mlp_output_dim=d_time_mlp_output_dim,
            condition_latent_dim=self.latent_dim, # Assuming condition_latent has self.latent_dim channels
            cond_mlp_output_dim=d_cond_mlp_output_dim,
            num_conv_blocks=d_num_conv_blocks_lddgan
        )
        
        self.lddgan_posterior_coeffs = None 

        self.lr_g = lr_g # Explicitly store lr_g
        self.lr_d_lddgan = lr_d_lddgan
        self.r1_gamma_lddgan = r1_gamma_lddgan
        self.gan_loss_weight_g = gan_loss_weight_g
        self.rec_loss_weight_g = rec_loss_weight_g

        # Remove or comment out old GAN attributes if they exist from previous versions
        # self.log_sigma_gan = ...
        # self.discriminator = ... (old NLayerDiscriminator)
        # self.adversarial_trainer = ...


    def on_train_start(self):
        if self.lddgan_posterior_coeffs is None and hasattr(self, 'device'):
            self.lddgan_posterior_coeffs = LDDGANPosteriorCoefficients(self.model.forward_process, self.device)

    @torch.no_grad()
    def forward(self, condition_img, *args, **kwargs): # For sampling/inference
        condition_latent = self.ae.encode(self.input_T(condition_img.to(self.device))).detach() * self.latent_scale_factor
        # self.model is DenoisingDiffusionConditionalProcess, its __call__ or sample method is used
        output_latent = self.model(condition_latent, *args, **kwargs) 
        output_latent = output_latent / self.latent_scale_factor
        return self.output_T(self.ae.decode(output_latent))
    
    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.model.parameters(), lr=self.lr_g) 
        opt_D_lddgan = torch.optim.Adam(self.discriminator_lddgan.parameters(), lr=self.lr_d_lddgan, betas=(0.5, 0.9)) # LDDGAN often uses different betas for D
        return [opt_G, opt_D_lddgan]
             
    def training_step(self, batch, batch_idx, optimizer_idx):
        condition_img, target_img = batch
        device = condition_img.device # Ensuring device consistency

        if self.lddgan_posterior_coeffs is None: 
            self.lddgan_posterior_coeffs = LDDGANPosteriorCoefficients(self.model.forward_process, device)
        
        with torch.no_grad():
            condition_latent_raw = self.ae.encode(self.input_T(condition_img)).detach() * self.latent_scale_factor
            real_latent_samples = self.ae.encode(self.input_T(target_img)).detach() * self.latent_scale_factor # x0_target
        
        b = real_latent_samples.size(0)

        # Prepare pooled condition latent for D (if spatial)
        if condition_latent_raw.ndim > 2 and self.discriminator_lddgan.condition_latent_dim:
            condition_latent_pooled_for_D = F.adaptive_avg_pool2d(condition_latent_raw, (1,1)).view(b, -1)
        else: # If already [B,D] or D does not use it
            condition_latent_pooled_for_D = condition_latent_raw.view(b, -1) if condition_latent_raw.ndim > 2 else condition_latent_raw


        # ── Generator Update (optimizer_idx == 0) ───────────────────────
        if optimizer_idx == 0:
            self.model.train() # Ensure UNet (Generator part) is in train mode for dropout/batchnorm
            self.discriminator_lddgan.eval() # Discriminator should be frozen

            # 1. Diffusion Loss (Standard for UNet)
            t_diff = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=device).long()
            x_t_diff, noise_target = self.model.forward_process(real_latent_samples, t_diff, return_noise=True)
            unet_input_diffusion = torch.cat([x_t_diff, condition_latent_raw], dim=1)
            noise_hat = self.model.model(unet_input_diffusion, t_diff) 
            diffusion_loss = self.model.loss_fn(noise_target, noise_hat)

            # 2. LDDGAN-style GAN Loss for Generator
            t_gan = torch.randint(0, self.model.forward_process.num_timesteps - 1, (b,), device=device).long()
            
            _gan_noise_g = torch.randn_like(real_latent_samples) # Ensure same noise for pair
            _, x_tp1_real_gan = ddpm_q_sample_pairs(self.model.forward_process, real_latent_samples, t_gan, noise=_gan_noise_g)
            
            unet_input_gan = torch.cat([x_tp1_real_gan.detach(), condition_latent_raw], dim=1)
            noise_pred_for_xtp1 = self.model.model(unet_input_gan, t_gan) 
            x0_pred_gan = self.model.forward_process.predict_start_from_noise(
                x_tp1_real_gan.detach(), t_gan, noise_pred_for_xtp1
            )
            x_t_fake_gan = lddgan_sample_posterior(
                self.lddgan_posterior_coeffs, x0_pred_gan, x_tp1_real_gan.detach(), t_gan
            )
            d_fake_output = self.discriminator_lddgan(
                x_t_fake_gan, t_gan, x_tp1_real_gan.detach(), condition_latent_pooled_for_D.detach()
            )
            gan_loss_g = F.softplus(-d_fake_output).mean()
            rec_loss_g = F.l1_loss(x0_pred_gan, real_latent_samples.detach())
            
            loss_G_total = diffusion_loss + \
                           self.gan_loss_weight_g * gan_loss_g + \
                           self.rec_loss_weight_g * rec_loss_g
            
            self.log_dict({
                "G_diff_loss": diffusion_loss.item(), "G_gan_loss": gan_loss_g.item(),
                "G_rec_loss": rec_loss_g.item(), "G_total_loss": loss_G_total.item(),
            }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss_G_total
    
        # ── Discriminator Update (optimizer_idx == 1) ──────────────────
        elif optimizer_idx == 1:
            self.discriminator_lddgan.train() # Ensure Discriminator is in train mode
            self.model.eval() # Generator (UNet part) should be frozen

            t_gan = torch.randint(0, self.model.forward_process.num_timesteps - 1, (b,), device=device).long()

            # Real samples for D
            _gan_noise_d_real = torch.randn_like(real_latent_samples)
            x_t_real_gan, x_tp1_real_gan = ddpm_q_sample_pairs(
                self.model.forward_process, real_latent_samples.detach(), t_gan, noise=_gan_noise_d_real
            )
            x_t_real_gan.requires_grad_(True) 
            d_real_output = self.discriminator_lddgan(
                x_t_real_gan, t_gan, x_tp1_real_gan.detach(), condition_latent_pooled_for_D.detach()
            )
            loss_D_real = F.softplus(-d_real_output).mean()
            
            grad_real, = torch.autograd.grad(outputs=d_real_output.sum(), inputs=x_t_real_gan, create_graph=True)
            r1_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            r1_penalty_term = (self.r1_gamma_lddgan / 2) * r1_penalty
            
            loss_D_real_w_penalty = loss_D_real + r1_penalty_term
            
            # Fake samples for D
            with torch.no_grad(): 
                unet_input_gan_d = torch.cat([x_tp1_real_gan.detach(), condition_latent_raw.detach()], dim=1)
                noise_pred_for_xtp1_d = self.model.model(unet_input_gan_d, t_gan)
                x0_pred_gan_d = self.model.forward_process.predict_start_from_noise(
                    x_tp1_real_gan.detach(), t_gan, noise_pred_for_xtp1_d
                )
                x_t_fake_gan_d = lddgan_sample_posterior(
                    self.lddgan_posterior_coeffs, x0_pred_gan_d, x_tp1_real_gan.detach(), t_gan
                )
            d_fake_output = self.discriminator_lddgan(
                x_t_fake_gan_d.detach(), t_gan, x_tp1_real_gan.detach(), condition_latent_pooled_for_D.detach()
            )
            loss_D_fake = F.softplus(d_fake_output).mean()
            
            loss_D_total = loss_D_real_w_penalty + loss_D_fake

            self.log_dict({
                "D_real_loss": loss_D_real.item(), "D_fake_loss": loss_D_fake.item(),
                "D_r1_penalty": r1_penalty.item(), "D_total_loss": loss_D_total.item(),
            }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss_D_total

    def validation_step(self, batch, batch_idx):
        condition_img, target_img = batch
        device = condition_img.device # ensure device

        with torch.no_grad():
            condition_latent = self.ae.encode(self.input_T(condition_img.to(device))).detach() * self.latent_scale_factor
            real_latent_samples = self.ae.encode(self.input_T(target_img.to(device))).detach() * self.latent_scale_factor
        
        b = real_latent_samples.shape[0]
        t_val = torch.randint(0, self.model.forward_process.num_timesteps, (b,), device=device).long()
        
        # Standard diffusion loss for validation
        x_t_val, noise_target_val = self.model.forward_process(real_latent_samples, t_val, return_noise=True)
        unet_input_val = torch.cat([x_t_val, condition_latent], dim=1)
        noise_hat_val = self.model.model(unet_input_val, t_val)
        val_diffusion_loss = self.model.loss_fn(noise_target_val, noise_hat_val)
        
        self.log('val_diffusion_loss', val_diffusion_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_diffusion_loss

# Removed old AdversarialTraining class
